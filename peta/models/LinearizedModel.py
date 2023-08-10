import abc
import logging
import os
from typing import Callable, List
from copy import deepcopy
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
from torch import Tensor

# from functorch import make_functional_with_buffers
from torch.func import functional_call, jvp


log = logging.getLogger(__name__)


# class FunctionalWithBuffers:
#     """
#     A wrapper class for a functional version of a model with buffers.

#     Args:
#         func (callable): The function to apply to the input tensor.
#         buffers (list): A list of buffers for the function.

#     Attributes:
#         func (callable): The function to apply to the input tensor.
#         buffers (list): A list of buffers for the function.
#     """

#     def __init__(self, func: Callable, buffers: List):
#         """
#         Initialize the FunctionalWithBuffers class.

#         Args:
#             func (callable): The function to apply to the input tensor.
#             buffers (list): A list of buffers for the function.
#         """
#         self.func = func
#         self.buffers = buffers

#     def __call__(self, params: List, x: Tensor) -> Tensor:
#         """
#         Apply the function to the input tensor.

#         Args:
#             params (list): A list of parameters for the function.
#             x (Tensor): The input tensor.

#         Returns:
#             Tensor: The output tensor.
#         """
#         return self.func(params, self.buffers, x)


# class LinearizedModel(nn.Module):
#     """Creates a linearized version of a nn.Module.

#     The linearized version of a model is a proper PyTorch model and can be
#     trained as any other nn.Module.

#     Args:
#         model (nn.Module): The model to linearize. The trainable parameters of
#             the linearized model will be initialized to the parameters of this
#             model.
#         init_model (nn.Module): A model of the same type as `model` containing
#             the parameters around which the model is initialized. If not
#             provided, `model` is used as the initialization model.
#     """

#     def __init__(self, model: nn.Module, init_model: nn.Module = None) -> None:
#         """Initializes the linearized model."""
#         super().__init__()
#         if init_model is None:
#             init_model = model

#         super(nn.Module, self).__setattr__("init_model", init_model)
#         super(nn.Module, self).__setattr__("model", model)

#         func0, params0, self.buffers0 = make_functional_with_buffers(
#             init_model.eval(), disable_autograd_tracking=True
#         )
#         self.func0 = lambda params, x: func0(params, self.buffers0, x)

#         _, params, _ = make_functional_with_buffers(
#             model, disable_autograd_tracking=False
#         )

#         self.params = nn.ParameterList(params)
#         self.params0 = nn.ParameterList(params0)
#         self._model_name = model.__class__.__name__

#         # The intial parameters are not trainable.
#         for p in self.params0:
#             p.requires_grad = False

#         # The params are.
#         # for p in self.params:
#         #     p.requires_grad = True

#     def __call__(self, *args, **kwargs) -> torch.Tensor:
#         """Computes the linearized model output using a first-order Taylor decomposition."""
#         dparams = [p - p0 for p, p0 in zip(self.params, self.params0)]
#         out, dp = jvp(
#             lambda param: self.func0(param, *args, **kwargs),
#             (tuple(self.params0),),
#             (tuple(dparams),),
#         )
#         return out + dp


def dict_params_to_tuple(dict_params: dict):
    return tuple(v for k, v in dict_params.items())


class LinearizedModelWraper(nn.Module):
    def __init__(self, model: nn.Module, init_model: nn.Module = None):
        super().__init__()
        self.model = model
        if init_model is None:
            init_model = model
        assert not hasattr(self, "params0")
        params0 = deepcopy(
            OrderedDict((k, v.detach()) for k, v in init_model.named_parameters())
        )
        self.params0_keys = params0.keys()
        self.params0_values = nn.ParameterList(params0.values())
        for p in self.params0_values:
            p.requires_grad_(False)

    def tuple_params_to_dict(self, tuple_params):
        assert len(tuple_params) == len(self.params0_keys)
        state_dict = {}
        for k, p in zip(self.params0_keys, tuple_params):
            state_dict[k] = p
        return state_dict

    def forward(self, *args, **kwargs):
        """Computes the linearized model output using a first-order Taylor decomposition."""
        params0 = tuple(self.params0_values)
        params = dict_params_to_tuple(OrderedDict(self.named_parameters()))
        dparams = tuple(p - p0 for p, p0 in zip(params, params0))
        out, dp = jvp(
            lambda *param: functional_call(
                self.model, self.tuple_params_to_dict(param), args, kwargs
            ),
            params0,
            dparams,
        )
        return out + dp
