# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
from flax import linen as nn


class AlexNetMultichipModel(nn.Module):
    """
    JAX multichip implementation of the AlexNet model originally introduced by
    Alex Krizhevsky in papers:
      - "ImageNet Classification with Deep Convolutional Neural Networks"
        (https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
      - "One weird trick for parallelizing convolutional neural networks"
        (https://arxiv.org/abs/1404.5997)
    """

    axis_name: str
    num_devices: int
    train_mode: bool
    param_dtype: Optional[
        Union[str, type[Any], jnp.dtype, jax._src.typing.SupportsDType, Any]
    ] = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # Utilizing data parallelism for the Convolutional layers since their
        # parameters tensors are too small so the overhead of inter-device
        # communication outweighs the benefit of tensor parallelism. We don't
        # need to explicitely state the parameters partitioning because
        # replication is assumed by default.

        # First feature extraction layer
        x = nn.Conv(
            features=64,
            kernel_size=(11, 11),
            strides=(4, 4),
            padding=0,
            param_dtype=self.param_dtype,
        )(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))

        # Second feature extraction layer
        x = nn.Conv(
            features=192,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding=2,
            param_dtype=self.param_dtype,
        )(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))

        # Third feature extraction layer
        x = nn.Conv(
            features=384,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=1,
            param_dtype=self.param_dtype,
        )(x)
        x = nn.relu(x)

        # Fourth feature extraction layer
        x = nn.Conv(
            features=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=1,
            param_dtype=self.param_dtype,
        )(x)
        x = nn.relu(x)

        # Fifth feature extraction layer
        x = nn.Conv(
            features=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=1,
            param_dtype=self.param_dtype,
        )(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))

        # Flatten
        x = x.reshape((x.shape[0], -1))

        # Gather features from all devices (sub-batches)
        x = jax.lax.all_gather(x, self.axis_name, tiled=True)

        # Utilizing tensor parallelism for Dense layers.
        dense_kernel_init = nn.with_partitioning(
            nn.linear.default_kernel_init,
            (None, self.axis_name),
        )
        dense_bias_init = nn.with_partitioning(
            nn.linear.initializers.ones_init(), (self.axis_name)
        )

        # First classifier layer
        x = nn.Dense(
            features=4096 // self.num_devices,
            param_dtype=self.param_dtype,
            kernel_init=dense_kernel_init,
            bias_init=dense_bias_init,
        )(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5)(x, deterministic=not self.train_mode)

        # Gather results from all devices
        x = jax.lax.all_gather(x, self.axis_name, axis=1, tiled=True)

        # Second classifier layer
        x = nn.Dense(
            features=4096 // self.num_devices,
            param_dtype=self.param_dtype,
            kernel_init=dense_kernel_init,
            bias_init=dense_bias_init,
        )(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5)(x, deterministic=not self.train_mode)

        # Gather results from all devices
        x = jax.lax.all_gather(x, self.axis_name, axis=1, tiled=True)

        # Third classifier layer
        x = nn.Dense(
            features=1000 // self.num_devices,
            param_dtype=self.param_dtype,
            kernel_init=dense_kernel_init,
            bias_init=dense_bias_init,
        )(x)

        # Gather results from all devices
        x = jax.lax.all_gather(x, self.axis_name, axis=1, tiled=True)

        # Calculate probabilities
        x = nn.softmax(x)

        return x
