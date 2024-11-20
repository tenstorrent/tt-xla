# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import jax
import jax.numpy as jnp
import flax

from infrastructure import verify_module


@pytest.mark.parametrize(
    "act_shape",  ## NHWC
    [
        (1, 32, 32, 32),
        (1, 32, 32, 64),
        (1, 32, 32, 128),
        (1, 32, 64, 32),
        (1, 32, 64, 64),
        (1, 32, 64, 128),
        (1, 32, 128, 32),
        (1, 32, 128, 64),
        (1, 32, 128, 128),
        (1, 64, 32, 32),
        (1, 64, 32, 64),
        (1, 64, 32, 128),
        (1, 64, 64, 32),
        (1, 64, 64, 64),
        (1, 64, 64, 128),
        (1, 64, 128, 32),
        (1, 64, 128, 64),
        (1, 64, 128, 128),
        (1, 128, 32, 32),
        (1, 128, 32, 64),
        (1, 128, 32, 128),
        (1, 128, 64, 32),
        (1, 128, 64, 64),
        (1, 128, 64, 128),
        (1, 128, 128, 32),
        (1, 128, 128, 64),
        (1, 128, 128, 128),
    ],
)
def test_maxpool2d(
    act_shape,
):
    def module_maxpool(img):
        return flax.linen.max_pool(
            img, window_shape=(2, 2), strides=(2, 2), padding=((0, 0), (0, 0))
        )

    verify_module(
        module_maxpool,
        [act_shape],
        required_pcc=0.95,
        required_atol=float("inf"),
        dtype=jnp.bfloat16,
    )


def test_resnet_maxpool2d():
    def module_resnet_maxpool(x):
        x = flax.linen.max_pool(
            x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1))
        )
        return x

    verify_module(
        module_resnet_maxpool,
        [(1, 112, 112, 64)],
        required_pcc=0.95,
        required_atol=float("inf"),
        dtype=jnp.bfloat16,
    )
