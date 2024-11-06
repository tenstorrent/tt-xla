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
        return flax.linen.max_pool(img, window_shape=(2, 2), strides=(2, 2), padding=((0, 0), (0, 0)))

    verify_module(module_maxpool, [act_shape], required_pcc=0.95, required_atol=float("inf"), dtype=jnp.bfloat16)

@pytest.mark.skip("AssertionError.")
def test_resnet_maxpool2d():
    # This maxpool doesnt work on its own because of the reshape that is inserted on its input
    # Issue: https://github.com/tenstorrent/tt-metal/issues/12866
    # It works with the conv on top since the output is already flattened.
    # In resnet, this is essentially the sequence that occurs. The only difference is that
    # there are a few eltwise ops in between.
    def module_resnet_maxpool(act, weights):
        x = jax.lax.conv_general_dilated(act, weights, [2, 2], ((3, 3), (3, 3)), dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
        x = flax.linen.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
        return x

    verify_module(module_resnet_maxpool, [(1, 224, 224, 3), (64, 3, 7, 7)], required_pcc=0.95, required_atol=float("inf"), dtype=jnp.bfloat16)
