"""This script finetunes a pretrained language model on a text-to-text dataset."""
import os
import logging
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import lightning.pytorch as pl
import peft
import torch
from datasets import DatasetDict, load_dataset
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator

import peta
from peta.utils import TimeIt, TitledLog
from peta.utils.logging.rich import pprint_yaml, setup_colorlogging
from peta.utils.ml.devices import num_devices

log = logging.getLogger(__name__)

# disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Seq2SeqLMModule(pl.LightningModule):
    def __init__(
        self,
        model: AutoModelForSeq2SeqLM,
        tokenizer: AutoTokenizer,
        optim_cfg: DictConfig,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optim_cfg = optim_cfg

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            Dict: Dictionary containing the optimizer and learning rate scheduler.
        """
        optim = {}
        if "optimizer" in self.optim_cfg:
            optim["optimizer"]: torch.optim.Optimizer = instantiate(
                self.optim_cfg["optimizer"],
                params=self.parameters(),
            )
        if "lr_scheduler" in self.optim_cfg:
            optim["lr_scheduler"]: torch.optim.lr_scheduler.LRScheduler = instantiate(
                self.optim_cfg["lr_scheduler"],
                optimizer=optim["optimizer"],
            )
        log.info(f"{'configure_optimizers':=^50}")
        log.info(optim)
        return optim

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("train/loss", loss)
        return loss


@hydra.main("config", "finetune_lm", None)
def main(cfg: DictConfig):
    setup_colorlogging(force=True)
    pprint_yaml(OmegaConf.to_yaml(cfg))
    if hasattr(cfg, "seed") and cfg.seed is not None:
        log.info(f"set seed to {cfg.seed}")
        L.seed_everything(cfg.seed)

    # load pretrained model and tokenizer
    with TitledLog("load pretrained model and tokenizer", log_fn=log.info):
        tokenizer = instantiate(cfg.model.tokenizer)

        model = instantiate(cfg.model.model)
        if cfg.peft.peft_config is not None:
            peft_config: peft.PeftConfig = instantiate(cfg.peft.peft_config)
            if hasattr(cfg.peft, "seed") and cfg.peft.seed is not None:
                log.info(f"set peft seed to {cfg.peft.seed}")
                L.seed_everything(cfg.peft.seed)
            model = peft.get_peft_model(model, peft_config)
            model.print_trainable_parameters()
        else:
            log.info(f"no peft config found, use full finetuning.")
        module = Seq2SeqLMModule(model, tokenizer, optim_cfg=cfg.optim)

    # load dataset
    with TitledLog("load datasets and dataloaders", log_fn=log.info):
        datasets: DatasetDict = instantiate(cfg.dataset.datasets)

        # convert the task to text-to-text format
        if hasattr(cfg.dataset, "preprocessor"):
            preprocessor = instantiate(
                cfg.dataset.preprocessor,
                tokenizer=tokenizer,
                tokenizer_kwargs=cfg.model.tokenizer_kwargs
                if hasattr(cfg.model, "tokenizer_kwargs")
                else None,
            )
            datasets = datasets.map(
                preprocessor,
                **cfg.dataset.map_kwargs if hasattr(cfg.dataset, "map_kwargs") else {},
            )
        assert (
            cfg.batch_size % num_devices(cfg.trainer.devices) == 0
        ), "batch_size must be divisible by the number of devices."
        batch_size = cfg.batch_size // num_devices(cfg.trainer.devices)
        train_loader = DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=default_data_collator,
        )

    trainer = pl.Trainer(**cfg.trainer)
    trainer.fit(module, train_loader)
    exit(0)


if __name__ == "__main__":
    main()
