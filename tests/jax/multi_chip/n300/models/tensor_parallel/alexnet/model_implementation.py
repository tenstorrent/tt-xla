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
        x = x.reshape((x.shape[0], -1))
        x = jax.lax.all_gather(x, self.axis_name, tiled=False)
        x = jax.lax.all_gather(x, self.axis_name, axis=1, tiled=True)

        return x
