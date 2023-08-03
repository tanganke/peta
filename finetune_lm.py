"""This script finetunes a pretrained language model on a text-to-text dataset."""
import logging
from pathlib import Path

import hydra
import lightning as L
import lightning.pytorch as pl
import peft
import torch
from datasets import load_dataset
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import peta
from peta.utils.logging.rich import pprint_yaml, setup_colorlogging

log = logging.getLogger(__name__)


@hydra.main("config", "finetune_lm", None)
def main(cfg: DictConfig):
    setup_colorlogging(force=True)
    pprint_yaml(OmegaConf.to_yaml(cfg))
    if hasattr(cfg, "seed") and cfg.seed is not None:
        L.seed_everything(cfg.seed)

    # load pretrained model and tokenizer
    tokenizer = instantiate(cfg.model.tokenizer)

    model = instantiate(cfg.model.model)
    if cfg.peft.peft_config is not None:
        peft_config: peft.PeftConfig = instantiate(cfg.peft.peft_config)
        if hasattr(cfg.peft.seed) and cfg.peft.seed is not None:
            L.seed_everything(cfg.peft.seed)
        model = peft.get_peft_model(model, peft_config)

    # load dataset
    datasets = instantiate(cfg.dataset.datasets)

    exit(0)


if __name__ == "__main__":
    main()
