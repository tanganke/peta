import abc
import logging
import os
from copy import deepcopy
from typing import Any, Optional, Callable, List
from functools import partial

import torch
from torch.func import functional_call, jvp
from functorch import make_functional_with_buffers
from torch import Tensor, nn


log = logging.getLogger(__name__)


class FunctionalWithBuffers:
    """
    A wrapper class for a functional version of a model with buffers.

    Args:
        func (callable): The function to apply to the input tensor.
        buffers (list): A list of buffers for the function.

    Attributes:
        func (callable): The function to apply to the input tensor.
        buffers (list): A list of buffers for the function.
    """

    def __init__(self, func: Callable, buffers: List):
        """
        Initialize the FunctionalWithBuffers class.

        Args:
            func (callable): The function to apply to the input tensor.
            buffers (list): A list of buffers for the function.
        """
        self.func = func
        self.buffers = buffers

    def __call__(self, params: List, x: Tensor) -> Tensor:
        """
        Apply the function to the input tensor.

        Args:
            params (list): A list of parameters for the function.
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        return self.func(params, self.buffers, x)


class LinearizedModel(nn.Module):
    """Creates a linearized version of a nn.Module.

    The linearized version of a model is a proper PyTorch model and can be
    trained as any other nn.Module.
    """

    def __init__(
        self,
        model: nn.Module,
        init_model: Optional[nn.Module] = None,
    ) -> None:
        R"""Initializes the linearized model.

        Args:
            model (nn.Module): The model to linearize. The trainable parameters of
                the linearized model will be initialized to the parameters of this
                model.
            init_model (nn.Module): A model of the same type as `model` containing
                the parameters around which the model is initialized. If not
                provided, `model` is used as the initialization model.
        """
        super().__init__()

        if init_model is None:
            init_model = deepcopy(model)
            init_model.eval()
            for p in init_model.parameters():
                p.requires_grad = False

        self.params0 = dict(init_model.named_parameters())
        self.model = model
        self.forward_call = partial(functional_call, module=model)

    def forward(self, x: Tensor) -> torch.Tensor:
        """Computes the linearized model output using a first-order Taylor decomposition."""
        params = dict(self.model.named_parameters())
        dparams = {name: params[name] - self.params0[name] for name in params}
        out, dp = jvp(
            lambda param: self.forward_call(param, x),
            (tuple(self.params0),),
            (tuple(dparams),),
        )
        return out + dp
