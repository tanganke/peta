import numpy as np
import torch
from scipy.stats import spearmanr as _spearmanr
from torch import Tensor


def _to_np(x: Tensor):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.array(x)


def spearmanr(pred: Tensor, target: Tensor):
    """
    Spearman's rank correlation coefficient.
    """
    pred = _to_np(pred)
    target = _to_np(target)
    return _spearmanr(pred, target)[0]
