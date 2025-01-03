# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import pytest
from infra import ComparisonConfig, random_tensor, run_op_test


@pytest.fixture
def comparison_config() -> ComparisonConfig:
    config = ComparisonConfig()
    config.disable_all()
    config.pcc.enable()
    config.pcc.required_pcc = 0.95
    return config


@pytest.mark.parametrize(
    ["img_shape", "kernel_shape"],
    [
        ((1, 256, 512), (1024, 256, 1)),
        ((1, 256, 256), (512, 256, 1)),
        ((1, 512, 256), (512, 512, 1)),
        ((1, 512, 512), (1024, 512, 1)),
    ],
)
def test_conv1d(
    img_shape: tuple, kernel_shape: tuple, comparison_config: ComparisonConfig
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
