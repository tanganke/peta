#!/usr/bin/env python3
"""This script evaluate a loaded language model on a text-to-text dataset."""
# %%
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

from finetune_lm import load_model_from_config


def load_finetuned_model(
    model: str,
    dataset: str,
    finetune_mode: str,
    version: int,
):
    log_dir: Path = (
        Path("logs") / model / dataset / finetune_mode / f"version_{version}"
    )
    config_path = log_dir / "config.yaml"
    cfg: DictConfig = OmegaConf.load(config_path)

    # load model from config
    with TitledLog("load pretrained model and tokenizer", log_fn=log.info):
        _return = load_model_from_config(cfg)
        tokenizer: AutoTokenizer = _return["tokenizer"]
        model: AutoModelForSeq2SeqLM | peft.PeftModel = _return["model"]

        # load checkpoint
        checkpoint_dir = log_dir / "checkpoints"
        checkpoints = os.listdir(checkpoint_dir)
        # get checkpoint files with `step=2000.pth` in its name
        checkpoints = list(filter(lambda x: "step=2000.pth" in x, checkpoints))
        assert len(checkpoints) == 1
        checkpoint = checkpoints[0]
        log.info(f"load checkpoint from {checkpoint}")

        # load trainable parameters
        state_dict = torch.load(checkpoint_dir / checkpoint, map_location="cpu")
        model.load_state_dict(state_dict["state_dict"], strict=False)
        model.eval()

    return {
        "config": cfg,
        "model": model,
        "tokenizer": tokenizer,
    }


# %%
cfg: DictConfig
tokenizer: AutoTokenizer
batch_size = 8

# load dataset
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


if "validation" in datasets:
    val_dataset = datasets["validation"]
elif "validataion_matched" in datasets:
    # mnli
    val_dataset = datasets["validataion_matched"]
else:
    raise KeyError(datasets.keys())
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=cfg.num_workers,
    shuffle=False,
    collate_fn=default_data_collator,
)
