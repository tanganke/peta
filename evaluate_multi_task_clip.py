from evaluate_single_task_clip import *
import functools
import itertools
import os
from typing import Dict, List, Tuple, Any
from torch import nn, Tensor
import numpy as np
from peta.tasks.arithmetic import *

log = logging.getLogger(__name__)

if __name__ == "__main__":
    fabric = L.Fabric(accelerator="cuda", devices=[0])
    fabric.launch()


def evaluate_accuracy_for_clip_vision_model(
    clip_vision_model, datamodule: pl.LightningDataModule
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
    text = [f"a photo of a {c}" for c in datamodule.classes]
    test_loader = fabric.setup_dataloaders(test_loaders[dataset_name])

    acc = evaluate_accuracy(
        clip_model=clip_model,
        clip_processor=clip_processor,
        text=text,
        test_loader=test_loader,
    )

    return acc


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
                clip_vision_model=model, datamodule=datamodules[dataset_name]
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
        pretrained_model=pretrained_clip_vision_models["standard"],
        task_vector_dict=finetuned_clip_vison_models_task_vectors["standard"],
        result_path_template="results/{MODEL_NAME}/fft_ties_merging_num-tasks={num_tasks}.csv",
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
            # "standard": evaluate_fft_average,
            # "lora": evaluate_lora_avg,
            # "l_lora": evaluate_l_lora_avg,
        },
        "task_arithmetic": {
            # "standard": evaluate_fft_task_arithmetic,
            # "lora": evaluate_lora_task_arithmetic,
            # "l_lora": evaluate_l_lora_task_arithmetic,
        },
        "ties_merging": {
            "standard": evaluate_fft_ties_merging,
            # "lora": evaluate_lora_ties_merging,
            # "l_lora": evaluate_l_lora_ties_merging,
        },
        "lorahub": {
            # "standard": evaluate_fft_lorahub,
            # "lora": evaluate_lora_lorahub,
            # "l_lora": evluate_l_lora_lorahub,
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