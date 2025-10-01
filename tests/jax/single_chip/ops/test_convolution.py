# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import pytest
from infra import (
    ComparisonConfig,
    random_tensor,
    run_op_test,
    run_op_test_with_random_inputs,
)
from utils import Category


# TODO investigate why conv has such poor precision.
@pytest.fixture
def comparison_config() -> ComparisonConfig:
    config = ComparisonConfig()
    config.disable_all()
    config.pcc.enable()
    # Low PCC tracked in https://github.com/tenstorrent/tt-xla/issues/662
    config.pcc.required_pcc = 0.92
    return config


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.lax.conv_general_dilated",
    shlo_op_name="stablehlo.convolution",
)
@pytest.mark.parametrize(
    ["img_shape", "kernel_shape"],
    [
        ((1, 256, 512), (1024, 256, 1)),
        ((1, 256, 256), (512, 256, 1)),
        ((1, 512, 256), (512, 512, 1)),
        ((1, 512, 512), (1024, 512, 1)),
    ],
    ids=lambda val: f"{val}",
)
def test_conv1d(
    img_shape: tuple,
    kernel_shape: tuple,
    comparison_config: ComparisonConfig,
):
    def conv1d(img, weights):
        return jax.lax.conv_general_dilated(
            lhs=img,
            rhs=weights,
            window_strides=(1,),
            padding=[(0, 0)],
            lhs_dilation=None,
            rhs_dilation=(1,),
            dimension_numbers=("NCW", "OIW", "NCW"),
            feature_group_count=1,
            batch_group_count=1,
        )

    img = random_tensor(img_shape, dtype="bfloat16")
    kernel = random_tensor(kernel_shape, dtype="bfloat16")

    run_op_test(conv1d, [img, kernel], comparison_config)


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
        (1, 256, 64, 56, 56, 1, 1, 1, 1, 0),
        (1, 64, 64, 56, 56, 1, 1, 1, 1, 0),
        (1, 64, 64, 56, 56, 3, 3, 1, 1, 1),
        (1, 64, 256, 56, 56, 1, 1, 1, 1, 0),
        (1, 512, 256, 56, 56, 1, 1, 2, 2, 0),
        (1, 128, 256, 56, 56, 1, 1, 2, 2, 0),
        (1, 128, 128, 28, 28, 3, 3, 1, 1, 1),
        (1, 512, 128, 28, 28, 1, 1, 1, 1, 0),
        (1, 128, 512, 28, 28, 1, 1, 1, 1, 0),
        (1, 1024, 512, 28, 28, 1, 1, 2, 2, 0),
        (1, 256, 512, 28, 28, 1, 1, 2, 2, 0),
        (1, 256, 256, 14, 14, 3, 3, 1, 1, 1),
        (1, 1024, 256, 14, 14, 1, 1, 1, 1, 0),
        (1, 256, 1024, 14, 14, 1, 1, 1, 1, 0),
        (1, 2048, 1024, 14, 14, 1, 1, 2, 2, 0),
        (1, 512, 1024, 14, 14, 1, 1, 2, 2, 0),
        (1, 512, 512, 7, 7, 3, 3, 1, 1, 1),
        (1, 2048, 512, 7, 7, 1, 1, 1, 1, 0),
        (1, 512, 2048, 7, 7, 1, 1, 1, 1, 0),
        # MISCELLANEOUS
        (1, 64, 16, 115, 115, 4, 4, 1, 1, 0),
        (1, 64, 64, 8, 8, 3, 3, 1, 1, 1),
        (1, 64, 64, 16, 16, 3, 3, 1, 1, 1),
        (1, 256, 256, 7, 7, 3, 3, 1, 1, 1),
        (1, 256, 64, 56, 56, 1, 1, 2, 2, 0),
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
    run_op_test_with_random_inputs(
        conv2d,
        [img_shape, kernel_shape],
        comparison_config=comparison_config,
        dtype="bfloat16",
    )
