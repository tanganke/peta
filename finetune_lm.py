#!/usr/bin/env python3
"""This script finetunes a pretrained language model on a text-to-text dataset."""
import logging
import os
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import lightning.pytorch as pl
import peft
import torch
from datasets import DatasetDict, load_dataset
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from peft.tuners.lora import LoraLayer
from torch import Tensor, nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator

import peta
from peta.models.LinearizedModel import LinearizedModelWraper
from peta.utils import TimeIt, TitledLog
from peta.utils.logging.rich import pprint_yaml, setup_colorlogging
from peta.utils.ml.devices import num_devices

log = logging.getLogger(__name__)

# disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Seq2SeqLMModule(pl.LightningModule):
    def __init__(
        self,
        model: AutoModelForSeq2SeqLM | peft.PeftModel,
        tokenizer: AutoTokenizer,
        optim_cfg: DictConfig,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optim_cfg = optim_cfg

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

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
        if self.trainer.is_global_zero:
            log.info(f"{'configure_optimizers':=^50}")
            log.info(optim)
        return optim

    def training_step(self, batch, batch_idx: int):
        outputs = self.forward(**batch)
        loss = outputs.loss
        self.log("train/loss", loss)
        return loss

    def save_trainable_parameters(self):
        if self.logger.log_dir is not None:
            # save trainable parameters
            ckpt_path = (
                Path(self.trainer.log_dir)
                / "checkpoints"
                / f"epoch={self.current_epoch}_step={self.global_step}.pth"
            )
            if not ckpt_path.parent.exists():
                Path.mkdir(ckpt_path.parent, exist_ok=True)
            state_dict = dict(
                (k, p) for k, p in self.model.named_parameters() if p.requires_grad
            )
            torch.save(state_dict, ckpt_path)

    def on_train_epoch_end(self) -> None:
        self.save_trainable_parameters()


def _get_submodules(model: nn.Module, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


def linearize_lora_model(model: nn.Module):
    for key, module in model.named_modules():
        if isinstance(module, LoraLayer) and isinstance(module, nn.Linear):
            log.debug(f"convert {key} to linearized lora layer")
            parent, target, target_name = _get_submodules(model, key)
            setattr(parent, target_name, LinearizedModelWraper(target))
    return model


def load_model_from_config(cfg: DictConfig):
    tokenizer = instantiate(cfg.model.tokenizer)

    model = instantiate(cfg.model.model)
    if cfg.peft.peft_config is not None:
        peft_config: peft.PeftConfig = instantiate(cfg.peft.peft_config)
        #  https://github.com/huggingface/peft/issues/567
        peft_config.target_modules = list(peft_config.target_modules)
        if hasattr(cfg.peft, "seed") and cfg.peft.seed is not None:
            log.info(f"set peft seed to {cfg.peft.seed}")
            L.seed_everything(cfg.peft.seed)
        model = peft.get_peft_model(model, peft_config)
        if cfg.model.linearize:
            model = linearize_lora_model(model)
        model.print_trainable_parameters()
    else:
        log.info(f"no peft config found, use full finetuning.")

    module = Seq2SeqLMModule(model, tokenizer, optim_cfg=cfg.optim)
    return dict(tokenizer=tokenizer, model=model, module=module)


@hydra.main("config", "finetune_lm", None)
def main(cfg: DictConfig):
    setup_colorlogging(force=True)

    if hasattr(cfg, "seed") and cfg.seed is not None:
        log.info(f"set seed to {cfg.seed}")
        L.seed_everything(cfg.seed)

    from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

    # check finetune_mode
    if cfg.peft.peft_config is not None:
        # use_lora = True
        if cfg.model.linearize:
            finetune_mode = "l_lora"
        else:
            finetune_mode = "lora"
    else:
        # use_lora = False
        finetune_mode = "standard"

    logger = TensorBoardLogger(
        Path("logs") / cfg.model.name / cfg.dataset.name, name=finetune_mode
    )
    trainer = pl.Trainer(**OmegaConf.to_container(cfg.trainer), logger=logger)
    if trainer.log_dir is not None and trainer.is_global_zero:
        log.info(f"log_dir: {trainer.log_dir}")
        os.makedirs(trainer.log_dir, exist_ok=True)
        OmegaConf.save(cfg, Path(trainer.log_dir) / "config.yaml")

    if trainer.is_global_zero:
        pprint_yaml(OmegaConf.to_yaml(cfg))
    # load pretrained model and tokenizer
    with TitledLog("load pretrained model and tokenizer", log_fn=log.info):
        _return = load_model_from_config(cfg)
        tokenizer: AutoTokenizer = _return["tokenizer"]
        model: AutoModelForSeq2SeqLM | peft.PeftModel = _return["model"]
        module: Seq2SeqLMModule = _return["module"]

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

        # create dataloaders
        assert (
            cfg.batch_size % num_devices(cfg.trainer.devices) == 0
        ), "batch_size must be divisible by the number of devices."
        batch_size = cfg.batch_size // num_devices(cfg.trainer.devices)
        train_loader = DataLoader(
            datasets["train"],
            batch_size=batch_size,
            num_workers=cfg.num_workers,
            shuffle=True,
            collate_fn=default_data_collator,
        )
        # if "validation" in datasets:
        #     val_dataset = datasets["validation"]
        # elif "validataion_matched" in datasets:
        #     # mnli
        #     val_dataset = datasets["validataion_matched"]
        # else:
        #     raise KeyError(datasets.keys())
        # val_loader = DataLoader(
        #     val_dataset,
        #     batch_size=batch_size,
        #     num_workers=cfg.num_workers,
        #     shuffle=False,
        #     collate_fn=default_data_collator,
        # )

    trainer.fit(
        module,
        train_dataloaders=train_loader,
        # val_dataloaders=val_loader,
    )

    if trainer.is_global_zero:
        module.save_trainable_parameters()
    exit(0)


if __name__ == "__main__":
    main()
