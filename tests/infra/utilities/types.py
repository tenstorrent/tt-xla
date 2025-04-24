# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import Union

import jax
import torch
from flax import linen, nnx
from jaxtyping import PyTree as jax_pytree
from torch.utils._pytree import PyTree as torch_pytree
from transformers import FlaxPreTrainedModel

# Convenience alias. Used to jointly represent tensors from different frameworks.
Tensor = Union[jax.Array, torch.Tensor]

# Convenience alias. Used to jointly represent models (commonly called NN modules) from
# different frameworks.
# NOTE nnx.Module is the newest API, linen.Module is legacy but it is used in some
# huggingface models.
Model = Union[nnx.Module, linen.Module, FlaxPreTrainedModel, torch.nn.Module]

# Convenience alias. Used to jointly represent physical HW/device from different
# frameworks.
Device = Union[jax.Device, torch.device]

# Convenience alias.
PyTree = Union[jax_pytree, torch_pytree]


class Framework(Enum):
    JAX = "jax"
    TORCH = "torch"

    def __str__(self) -> str:
        return self.value

    def from_model_type(model: Model) -> Framework:
        if isinstance(model, (nnx.Module, linen.Module, FlaxPreTrainedModel)):
            return Framework.JAX
        elif isinstance(model, torch.nn.Module):
            return Framework.TORCH
        else:
            raise TypeError(
                f"No supported framework for model of type {(type(model))}."
            )
