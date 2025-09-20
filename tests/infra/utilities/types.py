# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import Union, Dict, Tuple

import jax
import torch
from flax import linen, nnx
from jaxtyping import PyTree as jax_pytree
from torch.utils._pytree import PyTree as torch_pytree
from transformers import FlaxPreTrainedModel
from torch_xla.distributed.spmd import Mesh

# Convenience alias. Used to jointly represent tensors from different frameworks.
Tensor = Union[jax.Array, torch.Tensor]

# Convenience alias. Used to jointly represent models (commonly called NN modules) from
# different frameworks.
# NOTE nnx.Module is the newest API, linen.Module is legacy but it is used in all
# huggingface models.
Model = Union[nnx.Module, linen.Module, FlaxPreTrainedModel, torch.nn.Module]

# Convenience alias. Used to jointly represent physical HW/device from different
# frameworks.
Device = Union[jax.Device, torch.device]

# Convenience alias.
PyTree = Union[jax_pytree, torch_pytree]

Mesh = Mesh

ShardSpec = Dict[Tensor, Tuple[str, ...]]


class Framework(Enum):
    JAX = "jax"
    TORCH = "torch"

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other) -> bool:
        if isinstance(other, Framework):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return False

    def __hash__(self) -> int:
        return hash(self.value)
