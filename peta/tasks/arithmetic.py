import torch
from typing import Dict
from torch import Tensor
from typing import List


def state_dict_sub(a: Dict, b: Dict, strict: bool = True):
    """
    Returns the difference between two state dicts.

    Args:
        a (Dict): The first state dict.
        b (Dict): The second state dict.
        strict (bool): Whether to check if the keys of the two state dicts are the same.

    Returns:
        Dict: The difference between the two state dicts.
    """
    if strict:
        assert set(a.keys()) == set(b.keys())

    diff = {}
    for k in a:
        diff[k] = a[k] - b[k]
    return diff


def state_dict_add(a: Dict, b: Dict, strict: bool = True):
    """
    Returns the sum of two state dicts.

    Args:
        a (Dict): The first state dict.
        b (Dict): The second state dict.
        strict (bool): Whether to check if the keys of the two state dicts are the same.

    Returns:
        Dict: The sum of the two state dicts.
    """
    if strict:
        assert set(a.keys()) == set(b.keys())

    diff = {}
    for k in a:
        diff[k] = a[k] + b[k]
    return diff


def state_dict_mul(state_dict: Dict, scalar: float):
    """
    Returns the product of a state dict and a scalar.

    Args:
        state_dict (Dict): The state dict to be multiplied.
        scalar (float): The scalar to multiply the state dict with.

    Returns:
        Dict: The product of the state dict and the scalar.
    """
    diff = {}
    for k in state_dict:
        diff[k] = scalar * state_dict[k]
    return diff


def state_dict_interpolation(
    state_dicts: List[Dict[str, Tensor]], scalars: List[float]
):
    """
    Interpolates between a list of state dicts using a list of scalars.

    Args:
        state_dicts (List[Dict[str, Tensor]]): The list of state dicts to interpolate between.
        scalars (List[float]): The list of scalars to use for interpolation.

    Returns:
        Dict: The interpolated state dict.
    """
    assert len(state_dicts) == len(
        scalars
    ), "The number of state_dicts and scalars must be the same"
    assert len(state_dicts) > 0, "The number of state_dicts must be greater than 0"
    assert all(
        [len(state_dicts[0]) == len(state_dict) for state_dict in state_dicts]
    ), "All state_dicts must have the same number of keys"

    interpolated_state_dict = {}
    for key in state_dicts[0]:
        interpolated_state_dict[key] = torch.zeros_like(state_dicts[0][key])
        for state_dict, scalar in zip(state_dicts, scalars):
            interpolated_state_dict[key] += scalar * state_dict[key]
    return interpolated_state_dict
