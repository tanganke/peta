import functools
import itertools
import os
import random
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import numpy
import numpy as np
import peft
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from evaluate_single_task_clip import *
from peta.tasks.arithmetic import *

log = logging.getLogger(__name__)

if __name__ == "__main__":
    fabric = L.Fabric(accelerator="cuda", devices=[0])
    fabric.launch()


def skip_if_exist_in_results(dataset_names: List[str], results: Dict[str, List[Any]]):
    if len(results.keys()) == 0:
        return False
    results = pd.DataFrame(results)
    indices = None
    for dataset_idx, dataset_name in enumerate(dataset_names):
        if indices is None:
            indices = results[f"dataset:{dataset_idx}"] == dataset_name
        else:
            indices = indices & (results[f"dataset:{dataset_idx}"] == dataset_name)
        if indices.sum() == 0:
            return False
    log.info(f"skip {dataset_names}")
    return True


def evaluate_accuracy_for_clip_vision_model(
    clip_vision_model, test_loader: DataLoader, classes: List[str]
) -> float:
    # replace the `clip_model.vision_model` with `clip_vision_model`
    clip_processor, clip_model = load_clip_model(
        MODEL_NAME_OR_PATH, local_files_only=True
    )
    clip_model.vision_model = deepcopy(clip_vision_model)

    # setup fabric modules
    clip_model.vision_model = fabric.setup_module(clip_model.vision_model)
    clip_model.visual_projection = fabric.setup_module(clip_model.visual_projection)

    # compute text features
    text = [f"a photo of a {c}" for c in classes]
    test_loader = fabric.setup_dataloaders(test_loader)

    acc = evaluate_accuracy(
        clip_model=clip_model,
        clip_processor=clip_processor,
        text=text,
        test_loader=test_loader,
    )

    return acc


def evaluate_loss(
    *,
    clip_model: CLIPModel,
    clip_processor,
    text: List[str],
    test_loader,
) -> float:
    clip_model.eval()
    # precompute the text features
    text_input = clip_processor(text, return_tensors="pt", padding=True)
    text_embeds = clip_model.get_text_features(**text_input)

    loss, count = 0, 0

    with TitledLog("Evaluate accuracy", log_fn=log.info):
        for batch in tqdm(test_loader):
            images, labels = batch
            with torch.no_grad():
                image_embeds = clip_model.get_image_features(pixel_values=images)

                # normalized features
                image_embeds = image_embeds / image_embeds.norm(
                    p=2, dim=-1, keepdim=True
                )
                text_embeds = text_embeds.to(image_embeds.device)
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

                # cosine similarity as logits
                logit_scale = clip_model.logit_scale.exp().item()
                logits_per_text = (
                    torch.matmul(text_embeds, image_embeds.t()) * logit_scale
                )
                logits_per_image = logits_per_text.t()

                loss += F.cross_entropy(
                    logits_per_image, labels, reduction="sum"
                ).item()
                count += len(labels)

    return loss / count


def evaluate_loss_for_clip_vision_model(
    clip_vision_model, test_loader: DataLoader, classes: List[str]
) -> float:
    # replace the `clip_model.vision_model` with `clip_vision_model`
    clip_processor, clip_model = load_clip_model(
        MODEL_NAME_OR_PATH, local_files_only=True
    )
    clip_model.vision_model = deepcopy(clip_vision_model)

    # setup fabric modules
    clip_model.vision_model = fabric.setup_module(clip_model.vision_model)
    clip_model.visual_projection = fabric.setup_module(clip_model.visual_projection)

    # compute text features
    text = [f"a photo of a {c}" for c in classes]
    test_loader = fabric.setup_dataloaders(test_loader)

    loss = evaluate_loss(
        clip_model=clip_model,
        clip_processor=clip_processor,
        text=text,
        test_loader=test_loader,
    )

    return loss


def evaluate_simple_average_multi_task(
    *,
    finetune_mode: str = "standard",
    pretrained_model=None,
    task_vector_dict=None,
    result_path_template: str = "results/{MODEL_NAME}/fft_ties_merging_num-tasks={num_tasks}.csv",
):
    """
    evaluate simple average method on full-finetuned models.
    """
    for num_tasks in range(2, len(DATASET_NAMES) + 1):
        assert num_tasks >= 1, "num_tasks must be >= 1"

        # setup result path and check if result file already exists
        # if result file already exists, skip
        result_path = result_path_template.format(
            MODEL_NAME=MODEL_NAME, num_tasks=num_tasks
        )
        if os.path.exists(result_path):
            log.info(f"skip {result_path}")
            continue

        if os.path.exists(result_path + ".temp"):
            results = pd.read_csv(result_path + ".temp").to_dict("list")
        else:
            results = defaultdict(lambda: list())
        for dataset_names in itertools.combinations(DATASET_NAMES, num_tasks):
            if skip_if_exist_in_results(dataset_names, results):
                continue
            log.info(
                f"num_tasks: {num_tasks}, dataset_names: {dataset_names}, finetune_mode: {finetune_mode}"
            )

            # compute average task vector, and add it to the pretrained model
            avg_task_vector = state_dict_avg(
                [task_vector_dict[d] for d in dataset_names]
            )
            model: nn.Module = deepcopy(pretrained_model)
            assert set(avg_task_vector.keys()).issubset(model.state_dict().keys())
            avg_task_vector = state_dict_add(
                avg_task_vector, model.state_dict(), strict=False
            )
            model.load_state_dict(avg_task_vector, strict=False)

            for dataset_idx, dataset_name in enumerate(dataset_names):
                results[f"dataset:{dataset_idx}"].append(dataset_name)
            for dataset_name in DATASET_NAMES:
                log.info(f"evaluating on dataset: {dataset_name}")
                score = evaluate_accuracy_for_clip_vision_model(
                    clip_vision_model=model,
                    test_loader=test_loaders[dataset_name],
                    classes=datamodules[dataset_name].classes,
                )
                results[dataset_name].append(score)
            print(pd.DataFrame(results))

        results = pd.DataFrame(results)
        results.to_csv(result_path, index=False)


def evaluate_fft_average():
    evaluate_simple_average_multi_task(
        finetune_mode="standard",
        pretrained_model=pretrained_clip_vision_models["standard"],
        task_vector_dict=finetuned_clip_vison_models_task_vectors["standard"],
        result_path_template="results/{MODEL_NAME}/fft_average_num-tasks={num_tasks}.csv",
    )


def evaluate_lora_avg():
    evaluate_simple_average_multi_task(
        finetune_mode="lora",
        pretrained_model=pretrained_clip_vision_models["lora"],
        task_vector_dict=finetuned_clip_vison_models_task_vectors["lora"],
        result_path_template="results/{MODEL_NAME}/lora_average_num-tasks={num_tasks}.csv",
    )


def evaluate_l_lora_avg():
    evaluate_simple_average_multi_task(
        finetune_mode="l_lora",
        pretrained_model=pretrained_clip_vision_models["l_lora"],
        task_vector_dict=finetuned_clip_vison_models_task_vectors["l_lora"],
        result_path_template="results/{MODEL_NAME}/l_lora_average_num-tasks={num_tasks}.csv",
    )


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
    skip_val_datasets: List[str] = [],
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

        for dataset_idx, dataset_name in enumerate(dataset_names):
            results[f"dataset:{dataset_idx}"].append(dataset_name)
        for dataset_name in DATASET_NAMES:
            if dataset_name in skip_val_datasets:
                continue
            log.info(f"evaluating on dataset: {dataset_name}")
            score = evaluate_accuracy_for_clip_vision_model(
                clip_vision_model=model,
                test_loader=test_loaders[dataset_name],
                classes=datamodules[dataset_name].classes,
            )
            results[dataset_name].append(score)
        print(pd.DataFrame(results))

    return results


def evaluate_task_arithmetic_multi_task(
    *,
    finetune_mode: str = "standard",
    pretrained_model: nn.Module = None,
    task_vector_dict: Dict[str, Dict[str, Tensor]] = None,
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

        if os.path.exists(result_path + ".temp"):
            results = pd.read_csv(result_path + ".temp").to_dict("list")
        else:
            results = defaultdict(lambda: list())
        for dataset_names in itertools.combinations(DATASET_NAMES, num_tasks):
            if skip_if_exist_in_results(dataset_names, results):
                continue
            log.info(
                f"num_tasks: {num_tasks}, dataset_names: {dataset_names}, finetune_mode: {finetune_mode}"
            )
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
        pretrained_model=pretrained_clip_vision_models["standard"],
        task_vector_dict=finetuned_clip_vison_models_task_vectors["standard"],
        result_path_template="results/{MODEL_NAME}/fft_task_addition_num-tasks={num_tasks}.csv",
    )


def evaluate_lora_task_arithmetic():
    evaluate_task_arithmetic_multi_task(
        finetune_mode="lora",
        pretrained_model=pretrained_clip_vision_models["lora"],
        task_vector_dict=finetuned_clip_vison_models_task_vectors["lora"],
        result_path_template="results/{MODEL_NAME}/lora_task_addition_num-tasks={num_tasks}.csv",
    )


def evaluate_l_lora_task_arithmetic():
    evaluate_task_arithmetic_multi_task(
        finetune_mode="l_lora",
        pretrained_model=pretrained_clip_vision_models["l_lora"],
        task_vector_dict=finetuned_clip_vison_models_task_vectors["l_lora"],
        result_path_template="results/{MODEL_NAME}/l_lora_task_addition_num-tasks={num_tasks}.csv",
    )


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
        # This loop adds the name of each dataset to the results dictionary, using the format "dataset:{dataset_idx}" as the key. The dataset_idx variable is the index of the dataset in the dataset_names list.
        for dataset_idx, dataset_name in enumerate(dataset_names):
            results[f"dataset:{dataset_idx}"].append(dataset_name)
        # This loop evaluates the model on each dataset using the specified metric function. The DATASET_NAMES variable is a list of all dataset names, and the `metric_func` variable is a dictionary of metric functions, where each key is a dataset name and each value is a metric function.
        for dataset_name in DATASET_NAMES:
            log.info(f"evaluating on dataset: {dataset_name}")
            score = evaluate_accuracy_for_clip_vision_model(
                clip_vision_model=model,
                test_loader=test_loaders[dataset_name],
                classes=datamodules[dataset_name].classes,
            )
            results[dataset_name].append(score)
        print(pd.DataFrame(results))
    return results


def evaluate_ties_merging_multi_task(
    *,
    finetune_mode: str = "standard",
    pretrained_model=None,
    task_vector_dict=None,
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
    for num_tasks in range(2, len(DATASET_NAMES) + 1):
        assert num_tasks >= 1, "num_tasks must be >= 1"
        result_path = result_path_template.format(
            MODEL_NAME=MODEL_NAME, num_tasks=num_tasks
        )
        if os.path.exists(result_path):  # skip if already exists
            continue

        if os.path.exists(result_path + ".temp"):
            results = pd.read_csv(result_path + ".temp").to_dict("list")
        else:
            results = defaultdict(lambda: list())
        for dataset_names in itertools.combinations(DATASET_NAMES, num_tasks):
            if skip_if_exist_in_results(dataset_names, results):
                continue
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
        pretrained_model=pretrained_clip_vision_models["standard"],
        task_vector_dict=finetuned_clip_vison_models_task_vectors["standard"],
        result_path_template="results/{MODEL_NAME}/fft_ties_merging_num-tasks={num_tasks}.csv",
    )


def evaluate_lora_ties_merging():
    evaluate_ties_merging_multi_task(
        finetune_mode="lora",
        pretrained_model=pretrained_clip_vision_models["lora"],
        task_vector_dict=finetuned_clip_vison_models_task_vectors["lora"],
        result_path_template="results/{MODEL_NAME}/lora_ties_merging_num-tasks={num_tasks}.csv",
    )


def evaluate_l_lora_ties_merging():
    evaluate_ties_merging_multi_task(
        finetune_mode="l_lora",
        pretrained_model=pretrained_clip_vision_models["l_lora"],
        task_vector_dict=finetuned_clip_vison_models_task_vectors["l_lora"],
        result_path_template="results/{MODEL_NAME}/l_lora_ties_merging_num-tasks={num_tasks}.csv",
    )


def get_score(
    weights: List[float],
    model: peft.PeftModel,
    cache: Dict[str, Dict[str, Union[str, int, float]]],
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
    loss = 0
    for lora_module_name in cache.keys():
        loss += evaluate_loss_for_clip_vision_model(
            model,
            test_loaders[lora_module_name],
            datamodules[lora_module_name].classes,
        )
    # L1 regularization term
    metric_val = loss + get_regular(weights)

    return metric_val


def lorahub_learning(
    *,
    model: peft.PeftModel,
    lora_module_names: List[str],
    lora_state_dicts: List[Dict[str, Tensor]],
    seed: int = 42,
    max_inference_step: int = 40,
):
    from functools import partial

    from peta.lorahub.algorithm import (
        default_l1_regularization,
        get_final_weights,
        ng,
        set_peft_model_state_dict,
    )

    # copy the model and make cache
    model = deepcopy(model)
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
    final_lora = get_final_weights(
        recommendation.value,
        lora_module_names,
        cache,
    )

    # set the final weights
    model.load_state_dict(final_lora, strict=False)

    return recommendation.value, model


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

        if os.path.exists(result_path + ".temp"):
            results = pd.read_csv(result_path + ".temp").to_dict("list")
        else:
            results = defaultdict(lambda: list())
        for dataset_names in itertools.combinations(DATASET_NAMES, num_tasks):
            if skip_if_exist_in_results(dataset_names, results):
                continue
            log.info(
                f"num_tasks: {num_tasks}, finetune_mode: {finetune_mode}, datset_names: {dataset_names}"
            )

            _, model = lorahub_learning(
                model=pretrained_model,
                lora_module_names=dataset_names,
                lora_state_dicts=[
                    # NOTE pass lora modules here, not task vectors
                    state_dict_add(
                        pretrained_model.state_dict(),
                        task_vectors_as_dict[d],
                        strict=False,
                    )
                    for d in dataset_names
                ],
                seed=42,
                max_inference_step=40,
            )

            for dataset_idx, dataset_name in enumerate(dataset_names):
                results[f"dataset:{dataset_idx}"].append(dataset_name)
            for dataset_name in DATASET_NAMES:
                log.info(f"evaluating on dataset: {dataset_name}")
                score = evaluate_accuracy_for_clip_vision_model(
                    clip_vision_model=model,
                    test_loader=test_loaders[dataset_name],
                    classes=datamodules[dataset_name].classes,
                )
                results[dataset_name].append(score)
            # save intermediate results
            pd.DataFrame(results).to_csv(result_path + ".temp", index=False)
            print(pd.DataFrame(results))

        # save results to csv file
        results = pd.DataFrame(results)
        results.to_csv(result_path, index=False)


def evaluate_fft_lorahub():
    evaluate_lorahub(
        finetune_mode="standard",
        pretrained_model=pretrained_clip_vision_models["standard"],
        task_vectors_as_dict=finetuned_clip_vison_models_task_vectors["standard"],
        result_path_template="results/{MODEL_NAME}/fft_lorahub_num-tasks={num_tasks}.csv",
    )


def evaluate_lora_lorahub():
    evaluate_lorahub(
        finetune_mode="lora",
        pretrained_model=pretrained_clip_vision_models["lora"],
        task_vectors_as_dict=finetuned_clip_vison_models_task_vectors["lora"],
        result_path_template="results/{MODEL_NAME}/lora_lorahub_num-tasks={num_tasks}.csv",
    )


def evluate_l_lora_lorahub():
    evaluate_lorahub(
        finetune_mode="l_lora",
        pretrained_model=pretrained_clip_vision_models["l_lora"],
        task_vectors_as_dict=finetuned_clip_vison_models_task_vectors["l_lora"],
        result_path_template="results/{MODEL_NAME}/l_lora_lorahub_num-tasks={num_tasks}.csv",
    )


def parse_args():
    import argparse

    from peta.utils.args import verify_str_arg

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="simple average")
    parser.add_argument("--finetune_mode", type=str, default="standard")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    evaluate_functions = {
        "simple_average": {
            "standard": evaluate_fft_average,
            "lora": evaluate_lora_avg,
            "l_lora": evaluate_l_lora_avg,
        },
        "task_arithmetic": {
            "standard": evaluate_fft_task_arithmetic,
            "lora": evaluate_lora_task_arithmetic,
            "l_lora": evaluate_l_lora_task_arithmetic,
        },
        "ties_merging": {
            "standard": evaluate_fft_ties_merging,
            "lora": evaluate_lora_ties_merging,
            "l_lora": evaluate_l_lora_ties_merging,
        },
        "lorahub": {
            "standard": evaluate_fft_lorahub,
            "lora": evaluate_lora_lorahub,
            "l_lora": evluate_l_lora_lorahub,
        },
        "tangent_proj": {
            # "standard": evaluate_fft_tangent_project,
            # "lora": evaluate_lora_tangent_project,
            # "l_lora": evaluate_l_lora_tangent_project,
        },
        "greedy_task_arithmetic": {
            # "standard": evaluate_fft_task_arithmetic,
            # "lora": evaluate_lora_greedy_task_arithmetic,
            # "l_lora": evaluate_l_lora_task_arithmetic,
        },
    }

    evaluate_functions[args.method][args.finetune_mode]()
