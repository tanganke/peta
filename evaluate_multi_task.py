# %% import libraries
import functools
import itertools
import logging
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator

import peta
from peta.models.LinearizedModel import LinearizedModelWraper
from peta.tasks.arithmetic import *
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
    model_name: str,
    dataset: str,
    finetune_mode: str,
    version: int,
):
    log_dir: Path = (
        Path("logs") / model_name / dataset / finetune_mode / f"version_{version}"
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
    model_name: str,
    dataset_name: str,
    finetune_mode: str,
    version: int,
):
    log_dir: Path = (
        Path("logs") / model_name / dataset_name / finetune_mode / f"version_{version}"
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
    model_name: str, dataset: str, finetune_mode: str = "standard", version: int = 0
):
    model = load_pretrained_model(
        model_name,
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
        for pred in all_preds:
            f.write(pred + "\n")
    with open("temp/all_labels.txt", "w") as f:
        for label in all_labels:
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


def load_finetuned_state_dict(
    model_name: str,
    dataset_name: str,
    finetune_mode: str,
    version: int,
):
    with TitledLog("load pretrained model and tokenizer", log_fn=log.info):
        log_dir: Path = (
            Path("logs")
            / model_name
            / dataset_name
            / finetune_mode
            / f"version_{version}"
        )
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
        return state_dict


def version_of_max_accuracy(
    model_name: str,
    single_task_data: pd.DataFrame,
    dataset_name: str,
    method: str,
) -> str:
    """
    Given a model name, a dataset name, and a DataFrame containing single task data,
    returns the version of the model with the highest accuracy on the given dataset.

    Args:
    - model_name (str): the name of the model to search for
    - single_task_data (pd.DataFrame): a DataFrame containing single task data
    - dataset_name (str): the name of the dataset to search for

    Returns:
    - version (str): the version of the model with the highest accuracy on the given dataset
    """

    seleted_rows = single_task_data[
        [
            all(b)
            for b in zip(
                single_task_data["model"] == model_name,
                single_task_data["dataset"] == dataset_name,
                single_task_data["method"] == method,
            )
        ]
    ]
    max_row = seleted_rows.loc[seleted_rows["accuracy"].idxmax()]
    version = max_row["version"]
    return version


# %%
def load_task_vectors(model_name: str, dataset_names: list) -> tuple:
    """
    Load task vectors for a given model and dataset.

    Args:
    - model_name (str): The name of the model to load task vectors for.
    - dataset_names (list): A list of dataset names to load task vectors for.

    Returns:
    - fft_pretrained_model (AutoModelForSeq2SeqLM): The pretrained FFT model.
    - lora_pretrained_model (AutoModelForSeq2SeqLM): The pretrained LORA model.
    - l_lora_pretrained_model (AutoModelForSeq2SeqLM): The pretrained L-LORA model.
    - fft_task_vector (dict): A dictionary containing the FFT task vectors for each dataset.
    - lora_task_vector (dict): A dictionary containing the LORA task vectors for each dataset.
    - l_lora_task_vector (dict): A dictionary containing the L-LORA task vectors for each dataset.
    - val_loaders (dict): A dictionary containing the validation data loaders for each dataset.
    - tokenizer (AutoTokenizer): The tokenizer used for the model.
    """
    fft_state_dict: Dict[str, Dict[str, Tensor]] = {}
    lora_state_dict: Dict[str, Dict[str, Tensor]] = {}
    l_lora_state_dict: Dict[str, Dict[str, Tensor]] = {}

    fft_pretrained_model: AutoModelForSeq2SeqLM = None
    lora_pretrained_model: AutoModelForSeq2SeqLM = None
    l_lora_pretrained_model: AutoModelForSeq2SeqLM = None

    fft_task_vector: Dict[str, Tuple[int, Dict[str, Tensor]]] = {}
    lora_task_vector: Dict[str, Tuple[int, Dict[str, Tensor]]] = {}
    l_lora_task_vector: Dict[str, Tuple[int, Dict[str, Tensor]]] = {}

    val_loaders: Dict[str, DataLoader] = {}

    tokenizer: AutoTokenizer = None

    # load finetuned models, tokenizers and dataloaders
    single_task_data = pd.read_csv("results/flan-t5-base/single_task.csv")

    for dataset_name in dataset_names:
        # fft models
        # pick the version with highest accuracy
        # load state dict
        version = version_of_max_accuracy(
            model_name, single_task_data, dataset_name, method="fullfinetuned"
        )
        fft_state_dict[dataset_name] = load_finetuned_state_dict(
            model_name, dataset_name, "standard", version=version
        )

        pretrained_model, tokenizer, val_loader = load_pretrained_model_and_dataset(
            model_name, dataset_name, finetune_mode="standard", version=version
        )
        if fft_pretrained_model is None:
            fft_pretrained_model = pretrained_model
        # dataloader
        if dataset_name not in val_loaders:
            val_loaders[dataset_name] = fabric.setup_dataloaders(val_loader)

        fft_task_vector[dataset_name] = state_dict_sub(
            fft_state_dict[dataset_name],
            fft_pretrained_model.state_dict(),
            strict=False,
        )

        # lora models
        version = version_of_max_accuracy(
            model_name,
            single_task_data[single_task_data["LoRA.r"] == 16],
            dataset_name,
            "lora",
        )
        lora_state_dict[dataset_name] = load_finetuned_state_dict(
            model_name, dataset_name, "lora", version=version
        )

        if lora_pretrained_model is None:
            pretrained_model, _, val_loader = load_pretrained_model_and_dataset(
                model_name, dataset_name, finetune_mode="lora", version=version
            )
            lora_pretrained_model = pretrained_model
        lora_task_vector[dataset_name] = state_dict_sub(
            lora_state_dict[dataset_name],
            lora_pretrained_model.state_dict(),
            strict=False,
        )

        # l_lora model
        version = version_of_max_accuracy(
            model_name,
            single_task_data[single_task_data["LoRA.r"] == 16],
            dataset_name,
            "l_lora",
        )
        l_lora_state_dict[dataset_name] = load_finetuned_state_dict(
            model_name, dataset_name, "l_lora", version=version
        )

        if l_lora_pretrained_model is None:
            pretrained_model, _, val_loader = load_pretrained_model_and_dataset(
                model_name, dataset_name, finetune_mode="l_lora", version=version
            )
            l_lora_pretrained_model = pretrained_model
        l_lora_task_vector[dataset_name] = state_dict_sub(
            l_lora_state_dict[dataset_name],
            l_lora_pretrained_model.state_dict(),
            strict=False,
        )

    return (
        fft_pretrained_model,
        lora_pretrained_model,
        l_lora_pretrained_model,
        fft_task_vector,
        lora_task_vector,
        l_lora_task_vector,
        val_loaders,
        tokenizer,
    )


# %%
metric_func: Dict[str, Any] = defaultdict(lambda: evaluate_accuracy)
metric_func["glue-stsb"] = evaluate_spearman_rho

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

(
    fft_pretrained_model,
    lora_pretrained_model,
    l_lora_pretrained_model,
    fft_task_vector,
    lora_task_vector,
    l_lora_task_vector,
    val_loaders,
    tokenizer,
) = load_task_vectors(MODEL_NAME, DATASET_NAMES)


def get_task_vector(
    task_vectors: Dict[str, Dict[str, Tensor]], dataset_names: List[str]
) -> str:
    R"""
    Computes the task vector for the given dataset names.
        \tau_1 + \tau_2 + \tau_3 ... + \tau_n

    Args:
        task_vector (Dict[str, Dict[str, Tensor]]): A dictionary containing task vectors for each dataset.
        dataset_names (List[str]): A list of dataset names for which to compute the task vector.
    """
    task_vector = None
    for dataset_name in dataset_names:
        if task_vector is None:
            task_vector = task_vectors[dataset_name]
        else:
            task_vector = state_dict_add(task_vector, task_vectors[dataset_name])
    return task_vector


def evaluate_fft_task_addition():
    for num_tasks in reversed(range(0, len(DATASET_NAMES) + 1)):
        if os.path.exists(
            f"results/{MODEL_NAME}/fft_task_addition_num-tasks={num_tasks}.csv"
        ):  # skip if already exists
            continue
        finetune_mode = "standard"
        results = defaultdict(lambda: list())
        for dataset_names in itertools.combinations(DATASET_NAMES, num_tasks):
            task_vector = get_task_vector(fft_task_vector, dataset_names)
            for scaling_factor in np.linspace(0, 1, 21):
                results["scaling_factor"].append(scaling_factor)
                model: nn.Module = deepcopy(fft_pretrained_model)
                model.load_state_dict(
                    # \tau * \lambda + \theta_0
                    state_dict_add(
                        model.state_dict(),
                        state_dict_mul(task_vector, scaling_factor),
                        strict=False,
                    ),
                    strict=False,
                )
                model = fabric.setup_module(model)
                for dataset_idx, dataset_name in enumerate(dataset_names):
                    results[f"dataset:{dataset_idx}"].append(dataset_name)
                for dataset_name in DATASET_NAMES:
                    score = metric_func[dataset_name](
                        model, val_loaders[dataset_name], tokenizer
                    )
                    results[dataset_name].append(score)
                print(pd.DataFrame(results))

        results = pd.DataFrame(results)
        results.to_csv(
            f"results/{MODEL_NAME}/fft_task_addition_num-tasks={num_tasks}.csv",
            index=False,
        )


def evaluate_lora_task_addition():
    for num_tasks in reversed(range(0, len(DATASET_NAMES) + 1)):
        finetune_mode = "lora"
        if os.path.exists(
            f"results/{MODEL_NAME}/{finetune_mode}_task_addition_num-tasks={num_tasks}.csv",
        ):  # skip if already exists
            continue
        results = defaultdict(lambda: list())
        for dataset_names in itertools.combinations(DATASET_NAMES, num_tasks):
            task_vector = get_task_vector(lora_task_vector, dataset_names)
            for scaling_factor in np.linspace(0, 1, 21):
                results["scaling_factor"].append(scaling_factor)
                model: nn.Module = deepcopy(lora_pretrained_model)
                model.load_state_dict(
                    # \tau * \lambda + \theta_0
                    state_dict_add(
                        model.state_dict(),
                        state_dict_mul(task_vector, scaling_factor),
                        strict=False,
                    ),
                    strict=False,
                )
                model = fabric.setup_module(model)
                for dataset_idx, dataset_name in enumerate(dataset_names):
                    results[f"dataset:{dataset_idx}"].append(dataset_name)
                for dataset_name in DATASET_NAMES:
                    score = metric_func[dataset_name](
                        model, val_loaders[dataset_name], tokenizer
                    )
                    results[dataset_name].append(score)
                print(pd.DataFrame(results))

        results = pd.DataFrame(results)
        results.to_csv(
            f"results/{MODEL_NAME}/{finetune_mode}_task_addition_num-tasks={num_tasks}.csv",
            index=False,
        )


def evaluate_l_lora_task_addition():
    for num_tasks in reversed(range(0, len(DATASET_NAMES) + 1)):
        finetune_mode = "l_lora"
        if os.path.exists(
            f"results/{MODEL_NAME}/{finetune_mode}_task_addition_num-tasks={num_tasks}.csv",
        ):  # skip if already exists
            continue
        results = defaultdict(lambda: list())
        for dataset_names in itertools.combinations(DATASET_NAMES, num_tasks):
            task_vector = get_task_vector(l_lora_task_vector, dataset_names)
            for scaling_factor in np.linspace(0, 1, 21):
                results["scaling_factor"].append(scaling_factor)
                model: nn.Module = deepcopy(l_lora_pretrained_model)
                model.load_state_dict(
                    # \tau * \lambda + \theta_0
                    state_dict_add(
                        model.state_dict(),
                        state_dict_mul(task_vector, scaling_factor),
                        strict=False,
                    ),
                    strict=False,
                )
                model = fabric.setup_module(model)
                for dataset_idx, dataset_name in enumerate(dataset_names):
                    results[f"dataset:{dataset_idx}"].append(dataset_name)
                for dataset_name in DATASET_NAMES:
                    score = metric_func[dataset_name](
                        model, val_loaders[dataset_name], tokenizer
                    )
                    results[dataset_name].append(score)
                print(pd.DataFrame(results))

        results = pd.DataFrame(results)
        results.to_csv(
            f"results/{MODEL_NAME}/{finetune_mode}_task_addition_num-tasks={num_tasks}.csv",
            index=False,
        )


# %%

if __name__ == "__main__":
    evaluate_fft_task_addition()
    # evaluate_lora_task_addition()
    # evaluate_l_lora_task_addition()
