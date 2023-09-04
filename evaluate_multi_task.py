# %% import libraries
import functools
import itertools
import logging
import os
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import lightning as L
import lightning.pytorch as pl
import numpy
import numpy as np
import pandas as pd
import peft
import torch
from datasets import DatasetDict, load_dataset
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from peft.tuners.lora import LoraLayer
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
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
def _check_keys(state_dicts: List[Dict[str, Any]]):
    """
    Checks that the state dictionaries have the same keys.

    Args:
        state_dicts (List[Dict[str, Tensor]]): A list of dictionaries containing the state of PyTorch models.

    Raises:
        ValueError: If the state dictionaries have different keys.
    """
    # Get the keys of the first state dictionary in the list
    keys = set(state_dicts[0].keys())
    # Check that all the state dictionaries have the same keys
    for state_dict in state_dicts:
        assert keys == set(state_dict.keys()), "keys of state_dicts are not equal"


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
    for batch_idx, batch in zip(range(50), tqdm(val_loader)):
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


def load_validation_dataloader(cfg: DictConfig, tokenizer, batch_size=32):
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
        shuffle=True,  #! for efficiency, we shuffle the validation dataset, and validate on the first 50 batches.
        collate_fn=default_data_collator,
    )

    return val_loader


def evaluate_spearman_rho(model, val_loader: DataLoader, tokenizer):
    from tqdm import tqdm

    model = model.eval()
    all_preds: List[str] = []
    all_labels: List[str] = []
    for batch_idx, batch in zip(range(50), tqdm(val_loader)):
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
@torch.no_grad()
def load_task_vectors(model_name: str, dataset_names: list) -> tuple:
    """
    Load task vectors for a given model and dataset.

    Args:
        model_name (str): The name of the model to load task vectors for.
        dataset_names (list): A list of dataset names to load task vectors for.

    Returns:
        fft_pretrained_model (AutoModelForSeq2SeqLM): The pretrained FFT model.
        lora_pretrained_model (AutoModelForSeq2SeqLM): The pretrained LORA model.
        l_lora_pretrained_model (AutoModelForSeq2SeqLM): The pretrained L-LORA model.
        fft_task_vector (dict): A dictionary containing the FFT task vectors for each dataset.
        lora_task_vector (dict): A dictionary containing the LORA task vectors for each dataset.
        l_lora_task_vector (dict): A dictionary containing the L-LORA task vectors for each dataset.
        val_loaders (dict): A dictionary containing the validation data loaders for each dataset.
        tokenizer (AutoTokenizer): The tokenizer used for the model.
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
val_datasets = {d: val_loaders[d].dataset for d in DATASET_NAMES}


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


def evaluate_task_arithmetic(
    *,
    finetune_mode: str,
    pretrained_model: nn.Module,
    dataset_names: List[str],
    task_vector: Dict[str, Tensor],
    scaling_factors: List[float],
):
    num_tasks = len(dataset_names)
    # results is a dictionary-like object that is used to store the evaluation results for each task in the multi-task learning setup.
    #
    # pd.Dataframe(results):
    #      scaling_factor | dataset:0 | dataset:1 | ... | dataset:n | {DATASET_NAMES[0]} | {DATASET_NAMES[1]} | ... | {DATASET_NAMES[n]}
    results = defaultdict(lambda: list())

    for scaling_factor in scaling_factors:
        log.info(
            f"scaling_factor: {scaling_factor}, num_tasks: {num_tasks}, finetune_mode: {finetune_mode}, datset_names: {dataset_names}"
        )
        results["scaling_factor"].append(scaling_factor)
        model: nn.Module = deepcopy(pretrained_model)
        # check the the set of state dict keys is a subset of the model's state dict keys
        assert set(task_vector.keys()).issubset(
            model.state_dict().keys()
        ), "All task vectors must have corresponding parameters in the model"
        # Add the scaled task vector to the model's state dictionary using element-wise addition
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
            log.info(f"evaluating on dataset: {dataset_name}")
            score = metric_func[dataset_name](
                model, val_loaders[dataset_name], tokenizer
            )
            results[dataset_name].append(score)
        print(pd.DataFrame(results))

    return results


def evaluate_task_arithmetic_multi_task(
    *,
    finetune_mode: str = "standard",
    pretrained_model: nn.Module = fft_pretrained_model,
    task_vector_dict: Dict[str, Dict[str, Tensor]] = fft_task_vector,
    result_path_template: str = "results/{MODEL_NAME}/fft_task_addition_num-tasks={num_tasks}.csv",
):
    """
    Evaluates the multi-task model on all combinations of tasks with a given number of tasks, use task-arithmetic algorithm.

    Args:
        finetune_mode (str, optional): The finetune mode to use. Defaults to "standard".
        pretrained_model (nn.Module, optional): The pretrained model to use. Defaults to `fft_pretrained_model`.
        task_vector_dict (Dict[str, Dict[str, Tensor]], optional): A dictionary of task vectors for each task. Defaults to `fft_task_vector`.
        result_path_template (str, optional): The path template for the result file. Defaults to "results/{MODEL_NAME}/fft_task_addition_num-tasks={num_tasks}.csv".

    Returns:
        None
    """
    for num_tasks in range(2, len(DATASET_NAMES) + 1):
        assert num_tasks >= 1, "num_tasks must be >= 1"
        result_path = result_path_template.format(
            MODEL_NAME=MODEL_NAME, num_tasks=num_tasks
        )
        if os.path.exists(result_path):  # skip if already exists
            continue

        results = defaultdict(lambda: list())
        for dataset_names in itertools.combinations(DATASET_NAMES, num_tasks):
            task_vector = get_task_vector(task_vector_dict, dataset_names)
            _results = evaluate_task_arithmetic(
                finetune_mode=finetune_mode,
                pretrained_model=pretrained_model,
                dataset_names=dataset_names,
                task_vector=task_vector,
                scaling_factors=np.linspace(0, 1, 21),
            )
            for key in _results:
                results[key].extend(_results[key])
            # save intermediate results
            pd.DataFrame(results).to_csv(result_path + ".temp", index=False)

        results = pd.DataFrame(results)
        results.to_csv(result_path, index=False)


def evaluate_fft_task_arithmetic():
    evaluate_task_arithmetic_multi_task(
        finetune_mode="standard",
        pretrained_model=fft_pretrained_model,
        task_vector_dict=fft_task_vector,
        result_path_template="results/{MODEL_NAME}/fft_task_addition_num-tasks={num_tasks}.csv",
    )


def evaluate_lora_task_arithmetic():
    evaluate_task_arithmetic_multi_task(
        finetune_mode="lora",
        pretrained_model=lora_pretrained_model,
        task_vector_dict=lora_task_vector,
        result_path_template="results/{MODEL_NAME}/lora_task_addition_num-tasks={num_tasks}.csv",
    )


def evaluate_l_lora_task_arithmetic():
    evaluate_task_arithmetic_multi_task(
        finetune_mode="l_lora",
        pretrained_model=l_lora_pretrained_model,
        task_vector_dict=l_lora_task_vector,
        result_path_template="results/{MODEL_NAME}/l_lora_task_addition_num-tasks={num_tasks}.csv",
    )


def get_task_vector_p(
    task_vectors: Dict[str, Dict[str, Tensor]], dataset_names: List[str], p: float = 3
) -> str:
    R"""
    Computes the task vector for the given dataset names.
        \tau_1 + \tau_2 + \tau_3 ... + \tau_n

    Args:
        task_vector (Dict[str, Dict[str, Tensor]]): A dictionary containing task vectors for each dataset.
        dataset_names (List[str]): A list of dataset names for which to compute the task vector.
    """
    assert p != 0, "p must be != 0"
    task_vector = None
    for dataset_name in dataset_names:
        if task_vector is None:
            task_vector = state_dict_power(task_vectors[dataset_name], p)
        else:
            task_vector = state_dict_add(
                task_vector, state_dict_power(task_vectors[dataset_name], p)
            )
    return state_dict_power(task_vector, 1 / p)


def evaluate_l_lora_task_addition_p():
    for num_tasks in range(2, len(DATASET_NAMES) + 1):
        assert num_tasks >= 1, "num_tasks must be >= 1"
        finetune_mode = "l_lora"
        if os.path.exists(
            f"results/{MODEL_NAME}/{finetune_mode}_task_addition_num-tasks={num_tasks}.csv",
        ):  # skip if already exists
            continue
        results = defaultdict(lambda: list())
        for dataset_names in itertools.combinations(DATASET_NAMES, num_tasks):
            task_vector = get_task_vector_p(l_lora_task_vector, dataset_names)
            for scaling_factor in np.linspace(0, 1, 11):
                log.info(
                    f"scaling_factor: {scaling_factor}, num_tasks: {num_tasks}, finetune_mode: {finetune_mode}, datset_names: {dataset_names}"
                )
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
                    log.info(f"evaluating on dataset: {dataset_name}")
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


def evaluate_fft_task_addition_p():
    for num_tasks in range(2, len(DATASET_NAMES) + 1):
        assert num_tasks >= 1, "num_tasks must be >= 1"
        if os.path.exists(
            f"results/{MODEL_NAME}/fft_task_addition_num-tasks={num_tasks}.csv"
        ):  # skip if already exists
            continue
        finetune_mode = "standard"
        results = defaultdict(lambda: list())
        for dataset_names in itertools.combinations(DATASET_NAMES, num_tasks):
            task_vector = get_task_vector_p(fft_task_vector, dataset_names)
            for scaling_factor in np.linspace(0, 1, 11):
                log.info(
                    f"scaling_factor: {scaling_factor}, num_tasks: {num_tasks}, finetune_mode: {finetune_mode}, datset_names: {dataset_names}"
                )
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
                    log.info(f"evaluating on dataset: {dataset_name}")
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


def evaluate_lora_task_addition_p():
    for num_tasks in range(2, len(DATASET_NAMES) + 1):
        assert num_tasks >= 1, "num_tasks must be >= 1"
        finetune_mode = "lora"
        if os.path.exists(
            f"results/{MODEL_NAME}/{finetune_mode}_task_addition_num-tasks={num_tasks}.csv",
        ):  # skip if already exists
            continue
        results = defaultdict(lambda: list())
        for dataset_names in itertools.combinations(DATASET_NAMES, num_tasks):
            task_vector = get_task_vector_p(lora_task_vector, dataset_names)
            for scaling_factor in np.linspace(0, 1, 11):
                log.info(
                    f"scaling_factor: {scaling_factor}, num_tasks: {num_tasks}, finetune_mode: {finetune_mode}, datset_names: {dataset_names}"
                )
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
                    log.info(f"evaluating on dataset: {dataset_name}")
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


def evaluate_fft_average():
    """
    evaluate simple average method on full-finetuned models.
    """
    for num_tasks in range(2, len(DATASET_NAMES) + 1):
        assert num_tasks >= 1, "num_tasks must be >= 1"

        # setup result path and check if result file already exists
        # if result file already exists, skip
        result_path = f"results/{MODEL_NAME}/fft_average_num-tasks={num_tasks}.csv"
        if os.path.exists(result_path):
            log.info(f"skip {result_path}")
            continue

        finetune_mode = "standard"
        results = defaultdict(lambda: list())
        for dataset_names in itertools.combinations(DATASET_NAMES, num_tasks):
            log.info(
                f"num_tasks: {num_tasks}, dataset_names: {dataset_names}, finetune_mode: {finetune_mode}"
            )

            # compute average task vector, and add it to the pretrained model
            avg_task_vector = state_dict_avg(
                [fft_task_vector[d] for d in dataset_names]
            )
            model: nn.Module = deepcopy(fft_pretrained_model)
            avg_task_vector = state_dict_add(
                avg_task_vector, model.state_dict(), strict=False
            )
            model.load_state_dict(avg_task_vector, strict=False)

            model = fabric.setup_module(model)
            for dataset_idx, dataset_name in enumerate(dataset_names):
                results[f"dataset:{dataset_idx}"].append(dataset_name)
            for dataset_name in DATASET_NAMES:
                log.info(f"evaluating on dataset: {dataset_name}")
                score = metric_func[dataset_name](
                    model, val_loaders[dataset_name], tokenizer
                )
                results[dataset_name].append(score)
            print(pd.DataFrame(results))

        results = pd.DataFrame(results)
        results.to_csv(result_path, index=False)


def evaluate_lora_avg():
    for num_tasks in range(2, len(DATASET_NAMES) + 1):
        assert num_tasks >= 1, "num_tasks must be >= 1"

        # setup result path and check if result file already exists
        # if result file already exists, skip
        result_path = f"results/{MODEL_NAME}/lora_average_num-tasks={num_tasks}.csv"
        if os.path.exists(result_path):
            log.info(f"skip {result_path}")
            continue

        finetune_mode = "lora"
        results = defaultdict(lambda: list())
        for dataset_names in itertools.combinations(DATASET_NAMES, num_tasks):
            log.info(
                f"num_tasks: {num_tasks}, dataset_names: {dataset_names}, finetune_mode: {finetune_mode}"
            )

            avg_task_vector = state_dict_avg(
                [lora_task_vector[d] for d in dataset_names]
            )
            model: nn.Module = deepcopy(lora_pretrained_model)
            avg_task_vector = state_dict_add(
                avg_task_vector, model.state_dict(), strict=False
            )
            model.load_state_dict(avg_task_vector, strict=False)

            model = fabric.setup_module(model)
            for dataset_idx, dataset_name in enumerate(dataset_names):
                results[f"dataset:{dataset_idx}"].append(dataset_name)
            for dataset_name in DATASET_NAMES:
                log.info(f"evaluating on dataset: {dataset_name}")
                score = metric_func[dataset_name](
                    model, val_loaders[dataset_name], tokenizer
                )
                results[dataset_name].append(score)
            print(pd.DataFrame(results))

        results = pd.DataFrame(results)
        results.to_csv(result_path, index=False)


def evaluate_l_lora_avg():
    for num_tasks in range(2, len(DATASET_NAMES) + 1):
        assert num_tasks >= 1, "num_tasks must be >= 1"

        # setup result path and check if result file already exists
        # if result file already exists, skip
        result_path = f"results/{MODEL_NAME}/l_lora_average_num-tasks={num_tasks}.csv"
        if os.path.exists(result_path):
            log.info(f"skip {result_path}")
            continue

        finetune_mode = "l_lora"
        results = defaultdict(lambda: list())
        for dataset_names in itertools.combinations(DATASET_NAMES, num_tasks):
            log.info(
                f"num_tasks: {num_tasks}, dataset_names: {dataset_names}, finetune_mode: {finetune_mode}"
            )

            # compute average task vector, and add it to the pretrained model
            avg_task_vector = state_dict_avg(
                [l_lora_task_vector[d] for d in dataset_names]
            )
            model: nn.Module = deepcopy(l_lora_pretrained_model)
            avg_task_vector = state_dict_add(
                avg_task_vector, model.state_dict(), strict=False
            )
            model.load_state_dict(avg_task_vector, strict=False)

            model = fabric.setup_module(model)
            for dataset_idx, dataset_name in enumerate(dataset_names):
                results[f"dataset:{dataset_idx}"].append(dataset_name)
            for dataset_name in DATASET_NAMES:
                log.info(f"evaluating on dataset: {dataset_name}")
                score = metric_func[dataset_name](
                    model, val_loaders[dataset_name], tokenizer
                )
                results[dataset_name].append(score)
            print(pd.DataFrame(results))

        results = pd.DataFrame(results)
        results.to_csv(result_path, index=False)


def ties_merging(state_dicts: List[Dict[str, Tensor]], k: float):
    """
    Merges the state dictionaries of multiple PyTorch models using the TIES algorithm.

    Args:
        state_dicts (List[Dict[str, Tensor]]): A list of dictionaries containing the state of PyTorch models.
        k (float): The threshold for resetting the task checks. Should be a float between 0 and 1.

    Returns:
        Dict[str, Tensor]: A dictionary containing the merged state of the PyTorch models.
    """
    # Import the ties_merging module and check that the state dictionaries have the same keys
    import peta.tasks.ties_merging as tm

    _check_keys(state_dicts)

    # Convert the state dictionaries to vectors and merge them using the Ties-Merging algorithm
    task_vectors = torch.stack(tuple(map(tm.state_dict_to_vector, state_dicts)), dim=0)
    merged_task_vector = tm.ties_merging(task_vectors, k, merge_func="mean")

    # Convert the merged vector back to a state dictionary
    reference_state_dict = deepcopy(state_dicts[0])
    merged_state_dict = tm.vector_to_state_dict(
        merged_task_vector, reference_state_dict
    )

    return merged_state_dict


def evaluate_ties_merging(
    k: float,
    pretrained_model: nn.Module,
    task_vector_dict: Dict[str, Dict[str, Tensor]],
    dataset_names: List[str],
    finetune_mode: str,
    scaling_factors: List[float] = np.linspace(0, 1, 11),
):
    """
    Evaluates a PyTorch model on a set of tasks using the Ties-Merging algorithm.

    Args:
        k (float): The percentage of parameters to keep when computing the task vector.
        pretrained_model (nn.Module): The pre-trained PyTorch model to evaluate.
        task_vector_dict (Dict[str, Dict[str, Tensor]]): A dictionary of task vectors, where each key is a task name and each value is a dictionary of tensors representing the task vector.
        dataset_names (List[str]): A list of task names on which pretrained model finetuned.
        finetune_mode (str): The finetuning mode to use when scaling the model's weights.
        scaling_factors (List[float], optional): A list of scaling factors to use when scaling the merged task vector. Defaults to np.linspace(0, 1, 11).

    Returns:
        A dictionary containing the evaluation results for each task. The keys of the dictionary are the task names, and the
        values are lists of evaluation scores.
    """
    num_tasks = len(dataset_names)
    # results is a dictionary-like object that is used to store the evaluation results for each task in the multi-task learning setup.
    #
    # pd.Dataframe(results):
    #      scaling_factor | k | dataset:0 | dataset:1 | ... | dataset:n | {DATASET_NAMES[0]} | {DATASET_NAMES[1]} | ... | {DATASET_NAMES[n]}
    results = defaultdict(lambda: list())

    # the top-k% parameters are keeped
    task_vector = ties_merging([task_vector_dict[d] for d in dataset_names], k)
    # evaluate with a set of scaling factor hyperparameters
    for scaling_factor in scaling_factors:
        log.info(
            f"scaling_factor: {scaling_factor}, num_tasks: {num_tasks}, finetune_mode: {finetune_mode}, datset_names: {dataset_names}, k: {k}"
        )
        results["scaling_factor"].append(scaling_factor)
        results["k"].append(k)
        model: nn.Module = deepcopy(pretrained_model)
        # check the the set of state dict keys is a subset of the model's state dict keys
        assert set(task_vector.keys()).issubset(
            model.state_dict().keys()
        ), "All task vectors must have corresponding parameters in the model"
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
        # This loop adds the name of each dataset to the results dictionary, using the format "dataset:{dataset_idx}" as the key. The dataset_idx variable is the index of the dataset in the dataset_names list.
        for dataset_idx, dataset_name in enumerate(dataset_names):
            results[f"dataset:{dataset_idx}"].append(dataset_name)
        # This loop evaluates the model on each dataset using the specified metric function. The DATASET_NAMES variable is a list of all dataset names, and the `metric_func` variable is a dictionary of metric functions, where each key is a dataset name and each value is a metric function.
        for dataset_name in DATASET_NAMES:
            log.info(f"evaluating on dataset: {dataset_name}")
            score = metric_func[dataset_name](
                model, val_loaders[dataset_name], tokenizer
            )
            results[dataset_name].append(score)
        print(pd.DataFrame(results))
    return results


def evaluate_ties_merging_multi_task(
    *,
    finetune_mode: str = "standard",
    pretrained_model=fft_pretrained_model,
    task_vector_dict=fft_task_vector,
    result_path_template: str = "results/{MODEL_NAME}/fft_ties_merging_num-tasks={num_tasks}.csv",
):
    """
    Evaluates the multi-task arithmetic model on all combinations of tasks with a given number of tasks using the ties-merging method.

    Args:
        finetune_mode (str, optional): The finetune mode to use. Defaults to "standard".
        pretrained_model (nn.Module, optional): The pretrained model to use. Defaults to fft_pretrained_model.
        task_vector_dict (Dict[str, Dict[str, Tensor]], optional): A dictionary of task vectors for each task. Defaults to fft_task_vector.
        result_path_template (str, optional): The path template for the result file. Defaults to "results/{MODEL_NAME}/fft_ties_merging_num-tasks={num_tasks}.csv".

    Returns:
        None
    """
    for num_tasks in range(4, len(DATASET_NAMES) + 1):
        assert num_tasks >= 1, "num_tasks must be >= 1"
        result_path = result_path_template.format(
            MODEL_NAME=MODEL_NAME, num_tasks=num_tasks
        )
        if os.path.exists(result_path):  # skip if already exists
            continue

        results = defaultdict(lambda: list())
        for dataset_names in itertools.combinations(DATASET_NAMES, num_tasks):
            for k in [0.25, 0.5, 0.75, 1]:
                _results = evaluate_ties_merging(
                    k=k,
                    pretrained_model=pretrained_model,
                    task_vector_dict=task_vector_dict,
                    dataset_names=dataset_names,
                    finetune_mode=finetune_mode,
                    scaling_factors=np.linspace(0, 1, 5),
                )
                for key in _results:
                    results[key].extend(_results[key])
                # save intermediate results
                pd.DataFrame(results).to_csv(result_path + ".temp", index=False)

        results = pd.DataFrame(results)
        results.to_csv(result_path, index=False)


def evaluate_fft_ties_merging():
    evaluate_ties_merging_multi_task(
        finetune_mode="standard",
        pretrained_model=fft_pretrained_model,
        task_vector_dict=fft_task_vector,
        result_path_template="results/{MODEL_NAME}/fft_ties_merging_num-tasks={num_tasks}.csv",
    )


def evaluate_lora_ties_merging():
    evaluate_ties_merging_multi_task(
        finetune_mode="lora",
        pretrained_model=lora_pretrained_model,
        task_vector_dict=lora_task_vector,
        result_path_template="results/{MODEL_NAME}/lora_ties_merging_num-tasks={num_tasks}.csv",
    )


def evaluate_l_lora_ties_merging():
    evaluate_ties_merging_multi_task(
        finetune_mode="l_lora",
        pretrained_model=l_lora_pretrained_model,
        task_vector_dict=l_lora_task_vector,
        result_path_template="results/{MODEL_NAME}/l_lora_ties_merging_num-tasks={num_tasks}.csv",
    )


def lorahub_learning(
    *,
    model: peft.PeftModel,
    tokenizer: AutoTokenizer,
    lora_module_names: List[str],
    lora_state_dicts: List[Dict[str, Tensor]],
    dataset: Dataset,
    seed: int = 42,
    max_inference_step: int = 40,
):
    from functools import partial

    from peta.lorahub.algorithm import (
        default_get_loss,
        default_l1_regularization,
        get_final_weights,
        get_score,
        ng,
        set_peft_model_state_dict,
    )

    # copy the model and make cache
    model = deepcopy(model)
    model = fabric.setup_module(model)
    # Checks that the state dictionaries have the same keys.
    assert len(lora_module_names) == len(lora_state_dicts), "lengths are not equal"
    _check_keys(lora_state_dicts)

    cache = {
        name: state_dict
        for name, state_dict in zip(lora_module_names, lora_state_dicts)
    }

    # set seed for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)

    number_of_loras = len(lora_state_dicts)
    if number_of_loras == 0:
        print(
            "> No LoRA modules are provided. Please provide at least one LoRA module."
        )
        return None, None

    get_score_partial = partial(
        get_score,
        model=model,
        cache=cache,
        example_dataset=dataset,
        batch_size=32,
        get_loss=default_get_loss,
        get_regular=default_l1_regularization,
    )
    # set up the limit of the weights
    instrum = ng.p.Array(
        init=[0] * number_of_loras,
        upper=[1.5] * number_of_loras,
        lower=[-1.5] * number_of_loras,
    )
    optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=max_inference_step)
    log.info("> Begin to perform gradient-free optimization ...")
    recommendation = optimizer.minimize(get_score_partial, verbosity=1)
    final_lora = get_final_weights(recommendation.value, lora_module_names, cache)

    # set the final weights
    model.load_state_dict(final_lora, strict=False)

    return recommendation.value, model, tokenizer


def evaluate_lorahub(
    *,
    finetune_mode: str,
    pretrained_model: peft.PeftModel,
    task_vectors_as_dict: Dict[str, Dict[str, Tensor]],
    result_path_template: str,
):
    """
    Evaluates the LoraHub model on all combinations of tasks using a given finetune mode.

    Args:
        finetune_mode (str): The finetune mode to use.
        pretrained_model (peft.PeftModel): The pretrained model to use.
        task_vectors_as_dict (Dict[Dict[str, Tensor]]): A dictionary of task vectors for each task.
        result_path_template (str): The path template for the result file.

    Returns:
        None
    """
    # Iterate over all possible combinations of tasks
    for num_tasks in range(2, len(DATASET_NAMES) + 1):
        assert num_tasks >= 1, "num_tasks must be >= 1"
        result_path = result_path_template.format(
            MODEL_NAME=MODEL_NAME, num_tasks=num_tasks
        )
        if os.path.exists(result_path):  # skip if already exists
            continue

        results = defaultdict(lambda: list())
        for dataset_names in itertools.combinations(DATASET_NAMES, num_tasks):
            log.info(
                f"num_tasks: {num_tasks}, finetune_mode: {finetune_mode}, datset_names: {dataset_names}"
            )
            from torch.utils.data import ConcatDataset, Subset, random_split

            # truncate the dataset to 10*32 batches
            truncate_dataset = (
                lambda dataset: dataset
                if len(dataset) < 32 * 10
                else random_split(dataset, [32 * 10, len(dataset) - 32 * 10])[0]
            )
            dataset = ConcatDataset(
                truncate_dataset(val_datasets[d]) for d in dataset_names
            )
            _, model, _ = lorahub_learning(
                model=pretrained_model,
                tokenizer=tokenizer,
                lora_module_names=dataset_names,
                lora_state_dicts=[task_vectors_as_dict[d] for d in dataset_names],
                dataset=dataset,
                seed=42,
                max_inference_step=40,
            )

            for dataset_idx, dataset_name in enumerate(dataset_names):
                results[f"dataset:{dataset_idx}"].append(dataset_name)
            for dataset_name in DATASET_NAMES:
                log.info(f"evaluating on dataset: {dataset_name}")
                score = metric_func[dataset_name](
                    model, val_loaders[dataset_name], tokenizer
                )
                results[dataset_name].append(score)
            print(pd.DataFrame(results))


def evaluate_lora_lorahub():
    evaluate_lorahub(
        finetune_mode="lora",
        pretrained_model=lora_pretrained_model,
        task_vectors_as_dict=lora_task_vector,
        result_path_template="results/{MODEL_NAME}/lora_lorahub_num-tasks={num_tasks}.csv",
    )


def evluate_l_lora_lorahub():
    evaluate_lorahub(
        finetune_mode="l_lora",
        pretrained_model=l_lora_pretrained_model,
        task_vectors_as_dict=l_lora_task_vector,
        result_path_template="results/{MODEL_NAME}/l_lora_lorahub_num-tasks={num_tasks}.csv",
    )


# %%
def parse_args():
    import argparse

    from peta.utils.args import verify_str_arg

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="simple average")
    parser.add_argument("--finetune_mode", type=str, default="standard")
    args = parser.parse_args()

    verify_str_arg(
        args.method,
        "method",
        [
            "simple average",
            "task arithmetic",
            "ties merging",
            "lorahub",
        ],
    )
    verify_str_arg(
        args.finetune_mode,
        "finetune_mode",
        [
            "standard",
            "lora",
            "l_lora",
        ],
    )

    return args


if __name__ == "__main__":
    args = parse_args()

    evaluate_functions = {
        "simple average": {
            "standard": evaluate_fft_average,
            "lora": evaluate_lora_avg,
            "l_lora": evaluate_l_lora_avg,
        },
        "task arithmetic": {
            "standard": evaluate_fft_task_arithmetic,
            "lora": evaluate_lora_task_arithmetic,
            "l_lora": evaluate_l_lora_task_arithmetic,
        },
        "ties merging": {
            "standard": evaluate_fft_ties_merging,
            "lora": evaluate_lora_ties_merging,
            "l_lora": evaluate_l_lora_ties_merging,
        },
        "lorahub": {
            "lora": evaluate_lora_lorahub,
            "l_lora": evluate_l_lora_lorahub,
        },
    }

    evaluate_functions[args.method][args.finetune_mode]()
