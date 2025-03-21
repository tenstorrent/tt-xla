# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
from flax import nnx
from infra import random_tensor


class ExampleModel(nnx.Module):
    def __init__(self) -> None:
        w0_shape, w1_shape, b0_shape, b1_shape = (
            (784, 128),
            (128, 128),
            (1, 128),
            (1, 128),
        )

        self.w0 = random_tensor(w0_shape, minval=-0.01, maxval=0.01)
        self.w1 = random_tensor(w1_shape, minval=-0.01, maxval=0.01)
        self.b0 = random_tensor(b0_shape, minval=-0.01, maxval=0.01)
        self.b1 = random_tensor(b1_shape, minval=-0.01, maxval=0.01)

    def __call__(
        self, act: jax.Array, w0: jax.Array, b0: jax.Array, w1: jax.Array, b1: jax.Array
    ) -> jax.Array:
        # Note how activations, weights and biases are directly passed to the forward
        # method as inputs, `self` is not accessed. Otherwise they would be embedded
        # into jitted graph as constants.
        x = jnp.matmul(act, w0) + b0
        x = jnp.matmul(x, w1) + b1
        return x
