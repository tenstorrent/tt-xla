# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import pytest
from infra.comparison import ComparisonConfig
from infra.tester import run_op_test_with_random_inputs


@pytest.fixture
def comparison_config() -> ComparisonConfig:
    config = ComparisonConfig()
    config.disable_all()
    config.pcc.enable()
    return config


def add(x: jax.Array, y: jax.Array) -> jax.Array:
    return jax.numpy.add(x, y)


@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [
        [(32, 32), (32, 32)],
        [(64, 64), (64, 64)],
    ],
)
def test_add(x_shape: tuple, y_shape: tuple, comparison_config: ComparisonConfig):
    run_op_test_with_random_inputs(add, [x_shape, y_shape], comparison_config)
