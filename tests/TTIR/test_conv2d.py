# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import jax
import jax.numpy as jnp

from infrastructure import verify_module


@pytest.mark.parametrize(
    "batch_size, output_channels, input_channels, input_height, input_width, filter_height, filter_width, stride_h, stride_w, padding",
    (
        # RESNET
        (1, 64, 3, 224, 224, 7, 7, 2, 2, 3),
        (1, 256, 64, 56, 56, 1, 1, 1, 1, 0),
        (1, 64, 64, 56, 56, 1, 1, 1, 1, 0),
        (1, 64, 64, 56, 56, 3, 3, 1, 1, 1),
        (1, 64, 256, 56, 56, 1, 1, 1, 1, 0),
        (1, 512, 256, 56, 56, 1, 1, 2, 2, 0),
        (1, 128, 256, 56, 56, 1, 1, 2, 2, 0),
        (1, 128, 128, 28, 28, 3, 3, 1, 1, 1),
        (1, 512, 128, 28, 28, 1, 1, 1, 1, 0),
        (1, 128, 512, 28, 28, 1, 1, 1, 1, 0),
        # (1, 1024, 512, 28, 28, 1, 1, 2, 2, 0), Requires block sharding
        (1, 256, 512, 28, 28, 1, 1, 2, 2, 0),
        (1, 256, 256, 14, 14, 3, 3, 1, 1, 1),
        (1, 1024, 256, 14, 14, 1, 1, 1, 1, 0),
        (1, 256, 1024, 14, 14, 1, 1, 1, 1, 0),
        # (1, 2048, 1024, 14, 14, 1, 1, 2, 2, 0), Requires block sharding
        # (1, 512, 1024, 14, 14, 1, 1, 2, 2, 0), Requires block sharding
        # (1, 512, 512, 7, 7, 3, 3, 1, 1, 1), Requires block sharding
        # (1, 2048, 512, 7, 7, 1, 1, 1, 1, 0), Requires block sharding
        # (1, 512, 2048, 7, 7, 1, 1, 1, 1, 0), Requires block sharding

        # MISCELLANEOUS
        (1, 64, 16, 115, 115, 4, 4, 1, 1, 0),
        (1, 64, 64, 8, 8, 3, 3, 1, 1, 1),
        (1, 64, 64, 16, 16, 3, 3, 1, 1, 1),
        (1, 256, 256, 7, 7, 3, 3, 1, 1, 1),
        (1, 256, 64, 56, 56, 1, 1, 2, 2, 0), 
    ),
)
def test_conv2d(
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    filter_height,
    filter_width,
    stride_h,
    stride_w,
    padding
):
  def module_conv(img, weights):
    return jax.lax.conv_general_dilated(img, weights, [stride_h, stride_w], [[padding]*2]*2, dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
  

  img_shape = (batch_size, input_height, input_width, input_channels)
  weights_shape = (output_channels, input_channels, filter_height, filter_width)

  # Some resnet convolutions seem to require bfloat16, ttnn throws in runtime otherwise.
  # On another note, MaxPool2d is also only supported for bfloat16 in ttnn, so we have
  # to run resnet in bfloat16 for the time being.
  verify_module(module_conv, [img_shape, weights_shape], required_pcc=0.95, required_atol=float("inf"), dtype=jnp.bfloat16)
