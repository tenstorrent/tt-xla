# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Union

import jax

# Convenience alias. Could be used to represent jax.Array, torch.Tensor, np.ndarray, etc.
Tensor = Union[jax.Array]


class Framework(Enum):
    JAX = "jax"
    TORCH = "torch"
    NUMPY = "numpy"
