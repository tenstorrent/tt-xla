# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
from flax import linen as nn


class AlexNetModel(nn.Module):
    """
    JAX implementation of the AlexNet model originally introduced by Alex Krizhevsky
    in papers:
      - "ImageNet Classification with Deep Convolutional Neural Networks"
        (https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
      - "One weird trick for parallelizing convolutional neural networks"
        (https://arxiv.org/abs/1404.5997)
    """

    param_dtype: Optional[
        Union[str, type[Any], jnp.dtype, jax._src.typing.SupportsDType, Any]
    ] = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        # First feature extraction layer
        x = nn.Conv(
            features=64,
            kernel_size=(11, 11),
            strides=(4, 4),
            padding=0,
            param_dtype=self.param_dtype,
        )(x)
        x = nn.relu(x)
        x = LocalResponseNormalization()(x)
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
        x = LocalResponseNormalization()(x)
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

        # First classifier layer
        x = nn.Dense(features=4096, param_dtype=self.param_dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5)(x, deterministic=not train)

        # Second classifier layer
        x = nn.Dense(features=4096, param_dtype=self.param_dtype)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5)(x, deterministic=not train)

        # Third classifier layer
        x = nn.Dense(features=1000, param_dtype=self.param_dtype)(x)
        x = nn.softmax(x)

        return x


class LocalResponseNormalization(nn.Module):
    """Local response normalization layer per original paper implementation."""

    k: int = 2
    n: int = 5
    alpha: float = 1e-4
    beta: float = 0.75

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # Input is expected in (B, H, W, C) format
        num_channels = x.shape[-1]
        padded_x = jnp.pad(
            x, pad_width=[(0, 0), (0, 0), (0, 0), (self.n // 2, self.n // 2)]
        )

        def _apply_per_channel(c):
            window_sq = (
                jax.lax.dynamic_slice_in_dim(
                    operand=padded_x,
                    start_index=c,
                    slice_size=self.n,
                    axis=3,
                )
            ) ** 2
            window_sq_sum = (
                jnp.einsum("bhwc->bhw", window_sq) * self.alpha + self.k
            ) ** self.beta
            return x[:, :, :, num_channels] / window_sq_sum

        return jax.vmap(_apply_per_channel, in_axes=0, out_axes=3)(
            jnp.arange(num_channels)
        )
