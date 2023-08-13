# %% import libraries
import logging
import os
from pathlib import Path
from typing import Any, List

from tqdm import tqdm
import pandas as pd
import numpy as np
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


def load_validation_dataloaer(cfg: DictConfig, batch_size=32):
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


validataion_dataloaders = {}

# lora
for version in range(2, 10):
    lora_results = {"model": [], "dataset": [], "accuracy": [], "config": []}

    model_name = "flan-t5-base"
    finetune_mode = "lora"
    for dataset_name in [
        "glue-cola",
        "glue-mnli",
        "glue-mrpc",
        "glue-qqp",
        "glue-rte",
        "glue-sst2",
        "glue-stsb",
    ]:
        model = load_finetuned_model(model_name, dataset_name, finetune_mode, version)
        cfg, model, tokenizer = model["config"], model["model"], model["tokenizer"]

        if dataset_name in validataion_dataloaders:
            val_loader = validataion_dataloaders[dataset_name]
        else:
            val_loader = load_validation_dataloaer(cfg)
            validataion_dataloaders[dataset_name] = val_loader

        model = fabric.setup_module(model)
        val_loader = fabric.setup_dataloaders(val_loader)
        acc = evaluate_accuracy(model, val_loader, tokenizer)

        lora_results["model"].append(model_name)
        lora_results["dataset"].append(dataset_name)
        lora_results["accuracy"].append(acc)
        lora_results["config"].append(str(cfg))

        print(
            "model: {}, dataset: {}, accuracy: {}".format(model_name, dataset_name, acc)
        )

    lora_results = pd.DataFrame(lora_results)
    lora_results

    os.makedirs(f"results/{model_name}", exist_ok=True)
    lora_results.to_csv(f"results/{model_name}/lora_results_v{version}.csv")

# l_lora
for version in range(2, 12):
    l_lora_results = {"model": [], "dataset": [], "accuracy": [], "config": []}

    model_name = "flan-t5-base"
    finetune_mode = "l_lora"
    for dataset_name in [
        "glue-cola",
        "glue-mnli",
        "glue-mrpc",
        "glue-qqp",
        "glue-rte",
        "glue-sst2",
        "glue-stsb",
    ]:
        model = load_finetuned_model(
            model_name, dataset_name, finetune_mode, version
        )  # version 1
        cfg, model, tokenizer = model["config"], model["model"], model["tokenizer"]

        if dataset_name in validataion_dataloaders:
            val_loader = validataion_dataloaders[dataset_name]
        else:
            val_loader = load_validation_dataloaer(cfg)
            validataion_dataloaders[dataset_name] = val_loader

        model = fabric.setup_module(model)
        val_loader = fabric.setup_dataloaders(val_loader)
        acc = evaluate_accuracy(model, val_loader, tokenizer)

        l_lora_results["model"].append(model_name)
        l_lora_results["dataset"].append(dataset_name)
        l_lora_results["accuracy"].append(acc)
        l_lora_results["config"].append(cfg)

        print(
            "model: {}, dataset: {}, accuracy: {}".format(model_name, dataset_name, acc)
        )

    l_lora_results = pd.DataFrame(l_lora_results)
    l_lora_results

    os.makedirs(f"results/{model_name}", exist_ok=True)
    l_lora_results.to_csv(f"results/{model_name}/l_lora_results_v{version}.csv")
