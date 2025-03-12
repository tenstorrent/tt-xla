# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Union

import jax
from flax import linen, nnx
from transformers import FlaxPreTrainedModel

# Convenience alias. Could be used to represent jax.Array, torch.Tensor, np.ndarray, etc.
Tensor = Union[jax.Array]

# Convenience alias. Could be used to represent nnx.Module, torch.nn.Module, etc.
# NOTE nnx.Module is the newest API, linen.Module is legacy but it is used in some
# huggingface models.
Model = Union[nnx.Module, linen.Module, FlaxPreTrainedModel]


class Framework(Enum):
    JAX = "jax"
    TORCH = "torch"
    NUMPY = "numpy"

    def __str__(self) -> str:
        return self.value
