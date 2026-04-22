# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Tuple, Union

import torch
try:
    import jax
    from flax import linen, nnx
    from jaxtyping import PyTree as jax_pytree
except ImportError:
    jax = None
    linen = None
    nnx = None
    jax_pytree = Any
from torch.utils._pytree import PyTree as torch_pytree
try:
    from torch_xla.distributed.spmd import Mesh as TorchXLAMesh
except ImportError:
    TorchXLAMesh = Any

# Convenience alias. Used to jointly represent tensors from different frameworks.
Tensor = Union[(jax.Array if jax is not None else Any), torch.Tensor]

# Convenience alias. Used to jointly represent models (commonly called NN modules) from
# different frameworks.
# NOTE nnx.Module is the newest API, linen.Module is legacy but it is used in all
# huggingface models.
# NOTE FlaxPreTrainedModel was removed from transformers 5.x (Flax support dropped).
# EasyDel models (nnx.Module subclasses) and custom linen.Module models cover all JAX cases.
Model = Union[
    (nnx.Module if nnx is not None else Any),
    (linen.Module if linen is not None else Any),
    torch.nn.Module,
]

# Convenience alias. Used to jointly represent physical HW/device from different
# frameworks.
Device = Union[(jax.Device if jax is not None else Any), torch.device]

# Convenience alias.
PyTree = Union[jax_pytree, torch_pytree]

Mesh = TorchXLAMesh

ShardSpec = Dict[Tensor, Tuple[str, ...]]


class Framework(Enum):
    JAX = "jax"
    TORCH = "torch"
    TORCH_LLM = "torch_llm"

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
