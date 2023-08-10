import torch
from typing import Dict


def state_dict_sub(a: Dict, b: Dict, strict: bool = True):
    """
    Returns the difference between two state dicts.
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
    """
    diff = {}
    for k in state_dict:
        diff[k] = scalar * state_dict[k]
    return diff
