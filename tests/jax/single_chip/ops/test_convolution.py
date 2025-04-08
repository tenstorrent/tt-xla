# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import pytest
from infra import ComparisonConfig, random_tensor, run_op_test

from tests.utils import Category


# TODO investigate why conv has such poor precision.
@pytest.fixture
def comparison_config() -> ComparisonConfig:
    config = ComparisonConfig()
    config.disable_all()
    config.pcc.enable()
    config.pcc.required_pcc = 0.95
    return config


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.lax.conv_general_dilated",
    shlo_op_name="stablehlo.convolution",
)
@pytest.mark.parametrize(
    [
        "batch_size",
        "output_channels",
        "input_channels",
        "input_height",
        "input_width",
        "filter_height",
        "filter_width",
        "stride_h",
        "stride_w",
        "padding",
    ],
    [
        # RESNET
        (1, 64, 3, 224, 224, 7, 7, 2, 2, 3),
    ],
    ids=lambda val: f"{val}",
)
def test_conv2d(
    batch_size: int,
    output_channels: int,
    input_channels: int,
    input_height: int,
    input_width: int,
    filter_height: int,
    filter_width: int,
    stride_h: int,
    stride_w: int,
    padding: int,
    comparison_config: ComparisonConfig,
):
    def conv2d(img: jax.Array, kernel: jax.Array):
        return jax.lax.conv_general_dilated(
            img,
            kernel,
            [stride_h, stride_w],
            [[padding] * 2] * 2,
            dimension_numbers=("NHWC", "OIHW", "NHWC"),
        )

    img_shape = (batch_size, input_height, input_width, input_channels)
    kernel_shape = (output_channels, input_channels, filter_height, filter_width)

    # NOTE Some resnet convolutions seem to require bfloat16, ttnn throws in runtime
    # otherwise. On another note, MaxPool2d is also only supported for bfloat16 in ttnn,
    # so we have to run conv in bfloat16 for the time being.
    img = random_tensor(img_shape)
    kernel = random_tensor(kernel_shape)

    run_op_test(conv2d, [img, kernel], comparison_config)
