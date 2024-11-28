# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
from flax import nnx


class ExampleModel(nnx.Module):
    def __init__(self) -> None:
        w0_shape, w1_shape, b0_shape, b1_shape = (
            (784, 128),
            (128, 128),
            (1, 128),
            (1, 128),
        )

        self.w0 = jax.numpy.ones(w0_shape)
        self.w1 = jax.numpy.ones(w1_shape)
        self.b0 = jax.numpy.ones(b0_shape)
        self.b1 = jax.numpy.zeros(b1_shape)

    def __call__(
        self, act: jax.Array, w0: jax.Array, b0: jax.Array, w1: jax.Array, b1: jax.Array
    ) -> jax.Array:
        # Note how activations, weights and biases are directly passed to the forward
        # method. `self` is not accessed.
        x = jnp.matmul(act, w0) + b0
        x = jnp.matmul(x, w1) + b1
        return x
