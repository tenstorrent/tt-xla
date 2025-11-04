# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
from flax import nnx


class SimpleMLP(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(64, 128, rngs=rngs)
        self.linear2 = nnx.Linear(128, 10, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        return x
