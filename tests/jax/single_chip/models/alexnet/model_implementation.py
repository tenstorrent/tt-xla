# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
from flax import linen as nn
import torch
import numpy
import ml_dtypes
from jax import default_device
import jax


class AlexNetModel(nn.Module):
    """
    JAX implementation of the AlexNet model originally introduced by Alex Krizhevsky
    in papers:
      - "ImageNet Classification with Deep Convolutional Neural Networks"
      - "One weird trick for parallelizing convolutional neural networks"
    """

    param_dtype: Optional[
        Union[str, type[Any], jnp.dtype, jax._src.typing.SupportsDType, Any]
    ] = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.array, *, train: bool) -> jnp.array:
        # First feature extraction layer
        x = x.astype(jnp.bfloat16)
        conv1 = nn.Conv(
            features=64,
            kernel_size=(11, 11),
            strides=(4, 4),
            padding=0,
            param_dtype=self.param_dtype,
            name="conv1",
        )
        dummy_x = conv1(x)
        conv_param = self.get_variable('params', 'conv1')

        # Move input and parameters to CPU
        x_cpu = jax.device_put(x, jax.devices("cpu")[0])
        kernel_cpu = jax.device_put(conv_param['kernel'], jax.devices("cpu")[0])
        bias_cpu = jax.device_put(conv_param['bias'], jax.devices("cpu")[0])

        with default_device(jax.devices("cpu")[0]):
            x = conv1.apply({'params': {'kernel': kernel_cpu, 'bias': bias_cpu}}, x_cpu)
        x = jax.device_put(x, dummy_x.device)
        x = nn.relu(x)

        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))

        # Second feature extraction layer
        conv2 = nn.Conv(
            features=192,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding=2,
            param_dtype=self.param_dtype,
            name="conv2",
        )
        dummy_x = conv2(x)
        conv_param = self.get_variable('params', 'conv2')

        # Move input and parameters to CPU
        x_cpu = jax.device_put(x, jax.devices("cpu")[0])
        kernel_cpu = jax.device_put(conv_param['kernel'], jax.devices("cpu")[0])
        bias_cpu = jax.device_put(conv_param['bias'], jax.devices("cpu")[0])

        with default_device(jax.devices("cpu")[0]):
            x = conv2.apply({'params': {'kernel': kernel_cpu, 'bias': bias_cpu}}, x_cpu)
        x = jax.device_put(x, dummy_x.device)
        x = nn.relu(x)

        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))

        # Third feature extraction layer
        conv3 = nn.Conv(
            features=384,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=1,
            param_dtype=self.param_dtype,
            name="conv3",
        )
        dummy_x = conv3(x)
        conv_param = self.get_variable('params', 'conv3')

        # Move input and parameters to CPU
        x_cpu = jax.device_put(x, jax.devices("cpu")[0])
        kernel_cpu = jax.device_put(conv_param['kernel'], jax.devices("cpu")[0])
        bias_cpu = jax.device_put(conv_param['bias'], jax.devices("cpu")[0])

        with default_device(jax.devices("cpu")[0]):
            x = conv3.apply({'params': {'kernel': kernel_cpu, 'bias': bias_cpu}}, x_cpu)
        x = jax.device_put(x, dummy_x.device)
        x = nn.relu(x)

        # Fourth feature extraction layer
        conv4 = nn.Conv(
            features=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=1,
            param_dtype=self.param_dtype,
            name="conv4",
        )
        dummy_x = conv4(x)
        conv_param = self.get_variable('params', 'conv4')

        # Move input and parameters to CPU
        x_cpu = jax.device_put(x, jax.devices("cpu")[0])
        kernel_cpu = jax.device_put(conv_param['kernel'], jax.devices("cpu")[0])
        bias_cpu = jax.device_put(conv_param['bias'], jax.devices("cpu")[0])

        with default_device(jax.devices("cpu")[0]):
            x = conv4.apply({'params': {'kernel': kernel_cpu, 'bias': bias_cpu}}, x_cpu)
        x = jax.device_put(x, dummy_x.device)
        x = nn.relu(x)

        # Fifth feature extraction layer
        conv5 = nn.Conv(
            features=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=1,
            param_dtype=self.param_dtype,
            name="conv5",
        )
        dummy_x = conv5(x)
        conv_param = self.get_variable('params', 'conv5')

        # Move input and parameters to CPU
        x_cpu = jax.device_put(x, jax.devices("cpu")[0])
        kernel_cpu = jax.device_put(conv_param['kernel'], jax.devices("cpu")[0])
        bias_cpu = jax.device_put(conv_param['bias'], jax.devices("cpu")[0])

        with default_device(jax.devices("cpu")[0]):
            x = conv5.apply({'params': {'kernel': kernel_cpu, 'bias': bias_cpu}}, x_cpu)
        x = jax.device_put(x, dummy_x.device)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))

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
    def __call__(self, x: jnp.array) -> jnp.array:
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
