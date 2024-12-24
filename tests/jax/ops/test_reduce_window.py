# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import flax
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
    "img_shape",  ## NHWC
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
@pytest.mark.parametrize(
    ["window_shape", "strides", "padding"],
    [
        [(2, 2), (2, 2), ((0, 0), (0, 0))],
        # RESNET
        [(3, 3), (2, 2), ((1, 1), (1, 1))],
    ],
)
def test_reduce_window_max(
    img_shape: tuple,
    window_shape: tuple,
    strides: tuple,
    padding: tuple,
    comparison_config: ComparisonConfig,
):
    def maxpool2d(img: jax.Array):
        return flax.linen.max_pool(
            img, window_shape=window_shape, strides=strides, padding=padding
        )

    # NOTE Some resnet convolutions seem to require bfloat16, ttnn throws in runtime
    # otherwise. On another note, MaxPool2d is also only supported for bfloat16 in ttnn,
    # so we have to run conv in bfloat16 for the time being.
    # TODO raise an issue for this.
    img = random_tensor(img_shape, dtype="bfloat16")

    run_op_test(maxpool2d, [img], comparison_config)


# TODO add tests for reduce_window with add kernel.
