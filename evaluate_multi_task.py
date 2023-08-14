# %% import libraries
import functools
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import hydra
import lightning as L
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import peft
import torch
from datasets import DatasetDict, load_dataset
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from peft.tuners.lora import LoraLayer
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          default_data_collator)

import peta
from peta.models.LinearizedModel import LinearizedModelWraper
from peta.utils import TimeIt, TitledLog
from peta.utils.logging.rich import pprint_yaml, setup_colorlogging
from peta.utils.ml.devices import num_devices

log = logging.getLogger(__name__)

# disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from finetune_lm import load_model_from_config

# setup fabric
fabric = L.Fabric(accelerator="cuda", devices=[0])
fabric.launch()


# %% functions
def load_pretrained_model(
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
        model.eval()

    return {
        "config": cfg,
        "model": model,
        "tokenizer": tokenizer,
    }


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
        # assert (
        #     len(checkpoints) == 1
        # ), f"multiple checkpoints found, found checkpoints: {checkpoints}, checkpoint dir: {checkpoint_dir}"
        assert (
            len(checkpoints) >= 1
        ), f"no checkpoint found, checkpoint dir: {checkpoint_dir}"
        if len(checkpoints) > 1:
            log.warn(
                f"multiple checkpoints found, found checkpoints: {checkpoints}, checkpoint dir: {checkpoint_dir}"
            )
        checkpoint = checkpoints[0]
        log.info(f"load checkpoint from {checkpoint}")

        # load trainable parameters
        state_dict = torch.load(checkpoint_dir / checkpoint, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        model.eval()

    return {
        "config": cfg,
        "model": model,
        "tokenizer": tokenizer,
    }


def remove_special_tokens(tokenizer, token_list):
    ret = []
    for token in token_list:
        if token not in tokenizer.all_special_ids and token > 0:
            ret.append(token)
        if token == -100:
            break
    return ret


def evaluate_accuracy(model, val_loader: DataLoader, tokenizer):
    from tqdm import tqdm

    correct = 0
    total = 0

    model = model.eval()
    for batch_idx, batch in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            outputs = model.generate(batch["input_ids"])
            output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            labels = [remove_special_tokens(tokenizer, l) for l in batch["labels"]]
            labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # compare output_text and labels
            for i, j in zip(output_text, labels):
                if i == j:
                    correct += 1
                total += 1

    # return accuracy
    return correct / total


def load_pretrained_model_and_dataset(
    model: str, dataset: str, finetune_mode: str = "standard", version: int = 0
):
    model = load_pretrained_model(
        model,
        dataset,
        finetune_mode=finetune_mode,
        version=version,
    )
    cfg, model, tokenizer = model["config"], model["model"], model["tokenizer"]

    datasets = instantiate(cfg.dataset.datasets)

    if "validation" in datasets:
        val_dataset = datasets["validation"]
    elif "validation_matched" in datasets:
        # mnli
        val_dataset = datasets["validation_matched"]
    else:
        raise KeyError(datasets.keys())

    # convert the task to text-to-text format
    if hasattr(cfg.dataset, "preprocessor"):
        preprocessor = instantiate(
            cfg.dataset.preprocessor,
            tokenizer=tokenizer,
            tokenizer_kwargs=cfg.model.tokenizer_kwargs
            if hasattr(cfg.model, "tokenizer_kwargs")
            else None,
        )
        val_dataset = val_dataset.map(
            preprocessor,
            **cfg.dataset.map_kwargs if hasattr(cfg.dataset, "map_kwargs") else {},
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        num_workers=0,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    model = fabric.setup_module(model)
    val_loader = fabric.setup_dataloaders(val_loader)

    return model, tokenizer, val_loader


def load_validation_dataloaer(cfg: DictConfig, tokenizer, batch_size=32):
    datasets = instantiate(cfg.dataset.datasets)

    if "validation" in datasets:
        val_dataset = datasets["validation"]
    elif "validation_matched" in datasets:
        # mnli
        val_dataset = datasets["validation_matched"]
    else:
        raise KeyError(datasets.keys())

    # convert the task to text-to-text format
    if hasattr(cfg.dataset, "preprocessor"):
        preprocessor = instantiate(
            cfg.dataset.preprocessor,
            tokenizer=tokenizer,
            tokenizer_kwargs=cfg.model.tokenizer_kwargs
            if hasattr(cfg.model, "tokenizer_kwargs")
            else None,
        )
        val_dataset = val_dataset.map(
            preprocessor,
            **cfg.dataset.map_kwargs if hasattr(cfg.dataset, "map_kwargs") else {},
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    return val_loader


def evaluate_spearman_rho(model, val_loader: DataLoader, tokenizer):
    from tqdm import tqdm

    model = model.eval()
    all_preds: List[str] = []
    all_labels: List[str] = []
    for batch_idx, batch in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            outputs = model.generate(batch["input_ids"])
            output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            labels = [remove_special_tokens(tokenizer, l) for l in batch["labels"]]
            labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            all_preds.extend(output_text)
            all_labels.extend(labels)

    # save `all_preds` and `all_labels`
    with open("temp/all_preds.txt", "w") as f:
        for preds in all_preds:
            for pred in preds:
                f.write(pred + "\n")
    with open("temp/all_labels.txt", "w") as f:
        for labels in all_labels:
            for label in labels:
                f.write(label + "\n")

    # calculate spearman's rho
    # 1. convert string list `all_preds` and `all_labels` to numpy array
    # 2. compute spearman's rho
    from scipy.stats import spearmanr

    def parse_flost(s: str):
        import math

        try:
            return float(s)
        except:
            return 0.0

    all_preds = np.array([parse_flost(pred) for pred in all_preds])
    all_labels = np.array([parse_flost(label) for label in all_labels])
    rho = spearmanr(all_preds, all_labels)[0]
    return rho


# %%
fft_models: Dict[str, AutoModelForSeq2SeqLM] = {}
lora_models: Dict[str, AutoModelForSeq2SeqLM] = {}
l_lora_models: Dict[str, AutoModelForSeq2SeqLM] = {}
tokenizers: Dict[str, AutoTokenizer] = {}
val_loaders: Dict[str, DataLoader] = {}

MODEL_NAME = "flan-t5-base"
DATASET_NAMES = [
    "glue-cola",
    "glue-mnli",
    "glue-mrpc",
    "glue-qqp",
    "glue-rte",
    "glue-sst2",
    "glue-stsb",
]

# load finetuned models, tokenizers and dataloaders
for dataset_name in DATASET_NAMES:
    # fft model
    model = load_finetuned_model(MODEL_NAME, dataset_name, "standard", version=1)
    cfg, model, tokenizer = model["config"], model["model"], model["tokenizer"]
    fft_models[dataset_name] = model

    # lora model
    model = load_finetuned_model(MODEL_NAME, dataset_name, "lora", version=2)
    cfg, model, tokenizer = model["config"], model["model"], model["tokenizer"]
    lora_models[dataset_name] = model

    # l_lora model
    model = load_finetuned_model(MODEL_NAME, dataset_name, "l_lora", version=2)
    cfg, model, tokenizer = model["config"], model["model"], model["tokenizer"]
    l_lora_models[dataset_name] = model

    if dataset_name not in tokenizers:
        tokenizers[dataset_name] = tokenizer

    # dataloader
    if dataset_name not in val_loaders:
        val_loader = load_validation_dataloaer(cfg, tokenizer=tokenizer)
        val_loader = fabric.setup_dataloaders(val_loader)
        val_loaders[dataset_name] = val_loader

# %%
metric_func: Dict[str, Any] = defaultdict(lambda: evaluate_accuracy)
metric_func["glue-stsb"] = evaluate_spearman_rho

# %%
