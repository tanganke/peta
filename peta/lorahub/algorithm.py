import random
from functools import partial
from typing import Dict, List, Optional, Union

import nevergrad as ng
import numpy
import pandas as pd
import torch
from datasets import Dataset
from peft import PeftConfig, PeftModel
from peft.utils.save_and_load import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator
from torch import Tensor


def load_base_model_and_lora_modules(
    lora_module_list: List[str], model_name_or_path: Optional[str] = None
):
    """
    Loads a base model and a list of LoRA modules from the Hugging Face model hub.

    Args:
        lora_module_list (List[str]): A list of LoRA module names available in the Hugging Face model hub.
        model_name_or_path (Optional[str]): The name or path of the base model. If None, the base model is inferred from the first LoRA module in `lora_module_list`.

    Returns:
        Tuple[PeftModel, AutoTokenizer, Dict[str, Dict[str, Union[torch.Tensor, numpy.ndarray]]]]: A tuple containing the loaded PeftModel, the AutoTokenizer used for tokenization, and a dictionary of cached PeftModel state dictionaries for each LoRA module in `lora_module_list`.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load basic model
    default_peft_model_id = lora_module_list[0]
    # find the base model
    if model_name_or_path is None:
        model_name_or_path = PeftConfig.from_pretrained(
            default_peft_model_id
        ).base_model_name_or_path

    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # 0 is the default model
    try:
        peft_model = PeftModel.from_pretrained(base_model, default_peft_model_id)
    except:
        raise Exception(
            f"{default_peft_model_id} is unable to load into the model {model_name_or_path}"
        )

    peft_model = peft_model.to(device)
    peft_model.eval()

    print("> Begin to load lora modules")
    cache = {}

    first_dict = None

    for peft_model_id in tqdm(lora_module_list):
        print("> Loading {} ...".format(peft_model_id))
        cur_peft_model = PeftModel.from_pretrained(base_model, peft_model_id)
        cache[peft_model_id] = get_peft_model_state_dict(cur_peft_model)

        if first_dict is None:
            first_dict = cache[peft_model_id]
        # check whether the LoRA can be merged into one
        try:
            # detect whether the arch is the same
            for key in first_dict.keys():
                assert first_dict[key].shape == cache[peft_model_id][key].shape
        except:
            raise Exception(
                f"LoRA Modules {peft_model_id} cannot be merged since it has a different arch (e.g., rank)."
            )

    return peft_model, tokenizer, cache


def preprocess_function(examples, tokenizer):
    """
    standard preprocess function for dataset
    """
    inputs = examples["input"]
    targets = examples["output"]
    model_inputs = tokenizer(
        inputs,
        max_length=2048,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = tokenizer(
        targets,
        max_length=2048,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs


def load_dataset(
    example_inputs: List[str],
    example_outputs: Optional[List[str]],
    tokenizer: AutoTokenizer,
) -> Dataset:
    """
    Load a dataset from a list of input and output examples.

    Args:
    - example_inputs (List[str]): a list of input examples
    - example_outputs (Optional[List[str]]): a list of output examples. If None, an empty string is added for each input example.
    - tokenizer (AutoTokenizer): the tokenizer to use for preprocessing the dataset

    Returns:
    - Dataset: the processed dataset
    """
    # add empty string if example_outputs is None
    if example_outputs is None:
        example_outputs = [""] * len(example_inputs)
    df = [
        {"input": example_inputs[i], "output": example_outputs[i]}
        for i in range(len(example_inputs))
    ]
    dataset = Dataset.from_pandas(pd.DataFrame(df))
    preprocess_func_with_tokenizer = partial(preprocess_function, tokenizer=tokenizer)
    processed_datasets = dataset.map(
        preprocess_func_with_tokenizer,
        batched=True,
        num_proc=1,
        desc="Running tokenizer on dataset",
    )
    return processed_datasets


def default_get_loss(
    example_dataset: Dataset,
    model: PeftModel,
    batch_size: Optional[int],
    dataLoader_cls=DataLoader,
) -> float:
    """
    Get the loss of the model on the example dataset.
    Usually the example dataset only contains a few examples.

    Args:
    - example_dataset (Dataset): dataset containing examples to evaluate the model on
    - model (PeftModel): the model to evaluate
    - batch_size (Optional[int]): the batch size to use for evaluation. If None, use the length of the dataset.

    Returns:
    - float: the average loss over the number of examples in the dataset
    """
    data_batch_size = (
        len(example_dataset)
        if batch_size is None
        else min(len(example_dataset), batch_size)
    )

    # use gpu if available
    train_dataloader = dataLoader_cls(
        example_dataset,
        collate_fn=default_data_collator,
        batch_size=data_batch_size,
        pin_memory=True,
    )
    train_loss = 0
    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for _, batch in enumerate(tqdm(train_dataloader, "compute loss")):
            # to device
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss: Tensor = outputs.loss
            train_loss += loss.detach().float()
    loss = train_loss.float()
    # average loss over the number of examples
    return float(loss) / len(example_dataset)


def default_l1_regularization(weights: List[float]):
    """
    Get the L1 regularization term for the weights

    Args:
        weights (List[float]): the weights of the lora modules.
    """
    sum_of_squares = sum([abs(x) for x in weights]) / len(weights)
    return 0.05 * sum_of_squares


def get_score(
    weights: List[float],
    model: PeftModel,
    cache: Dict[str, Dict[str, Union[str, int, float]]],
    example_dataset: Dataset,
    batch_size: int,
    get_loss: partial,
    get_regular: partial,
) -> float:
    """
    Computes the score of the model with the given weights and configuration.

    Args:
        weights (List[float]): The weights of the model.
        model (PeftModel): The model to be evaluated.
        cache (Dict[str, Dict[str, Union[str, int, float]]]): The cache of the model.
        example_dataset (Dataset): The dataset to be used for evaluation.
        batch_size (int): The batch size to be used for evaluation.
        get_loss (partial): The function to be used to compute the loss.
        get_regular (partial): The function to be used to compute the regularization term.

    Returns:
        float: The score of the model with the given weights and configuration.
    """

    # the composed lora state dict
    final_state_dict = {}
    # module list is the list
    lora_module_list = list(cache.keys())
    # all keys are the same
    keys = cache[lora_module_list[0]].keys()
    for i, peft_model_id in enumerate(lora_module_list):
        lora_state_dict = cache[peft_model_id]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
    # check the the set of state dict keys is a subset of the model's state dict keys
    assert set(final_state_dict.keys()).issubset(
        model.state_dict().keys()
    ), "The set of state dict keys is not a subset of the model's state dict keys."
    # reload the model with the new adapter config
    # set_peft_model_state_dict(model, final_state_dict)
    model.load_state_dict(final_state_dict, strict=False)

    # minimize the metric
    loss = get_loss(example_dataset, model, batch_size)
    # L1 regularization term
    metric_val = loss + get_regular(weights)

    return metric_val

@torch.no_grad()
def get_final_weights(
    weights: List[float],
    lora_module_list: List[str],
    cache: Dict[str, Dict[str, Tensor]],
):
    """
    Computes the final state dictionary for a set of LORA modules using weighted averaging.

    Args:
        weights (List[float]): A list of weights for each LORA module. Should be a list of floats that sum to 1.
        lora_module_list (List[str]): A list of LORA module IDs.
        cache (Dict[str, Dict[str, Tensor]]): A dictionary-like object that stores cached values for a set of LORA modules.

    Returns:
        Dict[str, Tensor]: A dictionary containing the final state of the LORA modules, computed using weighted averaging.
    """
    # Initialize an empty dictionary for the final state dictionary
    final_state_dict = {}
    # Get the keys of the first LORA module in the list
    keys = cache[lora_module_list[0]].keys()
    # Compute the final state dictionary using weighted averaging
    for i, peft_model_id in enumerate(lora_module_list):
        lora_state_dict = cache[peft_model_id]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
    return final_state_dict


def lorahub_inference(
    example_inputs: List[str],
    model_or_name_path: Union[AutoModelForSeq2SeqLM, str],
    tokenizer_or_tokenizer_path: Union[AutoTokenizer, str],
    batch_size: int,
    # if not provided, we do not report the accuracy
    example_outputs: List[str] = None,
):
    def accuracy_score(outputs, ground_truths):
        correct = 0
        total = 0
        for output, truth in zip(outputs, ground_truths):
            if output.strip().lower().replace(".", "") == truth.strip().lower().replace(
                ".", ""
            ):
                correct += 1
            total += 1
        return correct / total * 100

    example_predictions = []
    # load model
    if isinstance(model_or_name_path, str):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_or_name_path)
    else:
        model = model_or_name_path

    # load tokenizer
    if isinstance(tokenizer_or_tokenizer_path, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_tokenizer_path)
    else:
        tokenizer = tokenizer_or_tokenizer_path

    # process dataset
    dataset = load_dataset(example_inputs, example_outputs, tokenizer)
    # use gpu if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i in range(0, len(dataset["input"]), batch_size):
        inputs = tokenizer(
            dataset["input"][i : i + batch_size],
            max_length=2048,
            return_tensors="pt",
            padding=True,
        ).to(device)
        outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=256)
        outputs = tokenizer.batch_decode(outputs.to("cpu"), skip_special_tokens=True)
        example_predictions.extend(outputs)

    if example_outputs is not None:
        print("Accuracy: ", accuracy_score(example_predictions, example_outputs))

    return example_predictions


def lorahub_learning(
    lora_module_list: List[str],
    example_inputs: List[str],
    example_outputs: List[str],
    max_inference_step: int,
    model_name_or_path=None,
    batch_size=None,
    get_loss=default_get_loss,
    get_regular=default_l1_regularization,
    seed=42,
):
    """
    Perform gradient-free optimization to learn the weights of a LoRA model.

    Args:
        lora_module_list (List[str]): List of paths to the LoRA modules.
        example_inputs (List[str]): List of input examples.
        example_outputs (List[str]): List of output examples.
        max_inference_step (int): Maximum number of iterations for the optimization.
        model_name_or_path (Optional[str]): Path or name of the base model to use.
        batch_size (Optional[int]): Batch size to use during inference.
        get_loss (Callable): Function to compute the loss.
        get_regular (Callable): Function to compute the regularization term.
        seed (int): Seed for reproducibility.

    Returns:
        Tuple: A tuple containing the learned weights, the trained model, and the tokenizer.
    """
    # set seed for reproducibility
    random.seed(seed)
    numpy.random.seed(seed)

    number_of_loras = len(lora_module_list)
    if number_of_loras == 0:
        print(
            "> No LoRA modules are provided. Please provide at least one LoRA module."
        )
        return None, None

    # load model
    model, tokenizer, cache = load_base_model_and_lora_modules(
        lora_module_list, model_name_or_path
    )
    # process dataset
    dataset = load_dataset(example_inputs, example_outputs, tokenizer)
    get_score_partial = partial(
        get_score,
        model=model,
        cache=cache,
        example_dataset=dataset,
        batch_size=batch_size,
        get_loss=get_loss,
        get_regular=get_regular,
    )
    # set up the limit of the weights
    instrum = ng.p.Array(
        init=[0] * number_of_loras,
        upper=[1.5] * number_of_loras,
        lower=[-1.5] * number_of_loras,
    )
    optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=max_inference_step)
    print("> Begin to perform gradient-free optimization ...")
    recommendation = optimizer.minimize(get_score_partial, verbosity=1)
    final_lora = get_final_weights(recommendation.value, lora_module_list, cache)
    # set the final weights
    set_peft_model_state_dict(model, final_lora)
    model = model.merge_and_unload()
    return recommendation.value, model, tokenizer
