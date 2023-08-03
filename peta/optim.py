from typing import Tuple

import numpy as np
import torch


def _warmup_lr(base_lr: float, warmup_length: int, step_idx: int):
    return base_lr * (step_idx + 1) / warmup_length


def _cos_lr(base_lr: float, max_steps: int, step_idx: int):
    lr = 0.5 * (1 + np.cos(np.pi * step_idx / max_steps)) * base_lr
    return lr


class CosineAnnealingWithWarmup:
    R"""
    a `max_steps`-step cosine annealing learning rate schedule with `warmup_steps` warm-up steps.
    The `step(step_idx)` method should be called every update step.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lrs: float | Tuple[float],
        warmup_steps: int,
        max_steps: int,
    ):
        super().__init__()
        self.optimizer = optimizer
        if isinstance(base_lrs, (float, int)):
            base_lrs = tuple(base_lrs for _ in optimizer.param_groups)
        self.base_lrs = base_lrs
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lrs(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self, step_idx: int = 0):
        warmup_length = self.warmup_steps
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if step_idx < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step_idx)
            else:
                lr = _cos_lr(base_lr, self.max_steps - warmup_length, step_idx - warmup_length)
            param_group["lr"] = lr  # assign learning rate

        self._last_lr = [param_group["lr"] for param_group in self.optimizer.param_groups]
