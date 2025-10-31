# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
from flax import linen as nn


class MNISTMLPModel(nn.Module):
    """MNIST MLP model implementation."""

    hidden_sizes: tuple[int]

    @nn.compact
    def __call__(self, x: jax.Array):
        x = x.reshape((x.shape[0], -1))

        for h in self.hidden_sizes:
            x = nn.Dense(features=h)(x)
            x = nn.relu(x)

        x = nn.Dense(features=10)(x)
        x = nn.softmax(x)

        return x


class MNISTMLPMultichipModel(nn.Module):
    """MNIST MLP multichip model implementation with tensor parallelism."""

    hidden_sizes: tuple[int]
    axis_name: str
    num_devices: int
    train_mode: bool
    param_dtype: Optional[
        Union[str, type[Any], jnp.dtype, jax._src.typing.SupportsDType, Any]
    ] = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))

        # Utilizing tensor parallelism for large Dense layers.
        dense_kernel_init = nn.with_partitioning(
            nn.linear.default_kernel_init,
            (None, self.axis_name),
        )
        dense_bias_init = nn.with_partitioning(
            nn.linear.initializers.ones_init(), (self.axis_name)
        )

        for hidden_size in self.hidden_sizes:
            x = nn.Dense(
                features=hidden_size // self.num_devices,
                param_dtype=self.param_dtype,
                kernel_init=dense_kernel_init,
                bias_init=dense_bias_init,
            )(x)
            x = nn.relu(x)
            # Gather results from all devices
            x = jax.lax.all_gather(x, self.axis_name, axis=1, tiled=True)

        # Final layer is pretty small so no point in parallelizing it.
        x = nn.Dense(features=10)(x)
        x = nn.softmax(x)

        return x
