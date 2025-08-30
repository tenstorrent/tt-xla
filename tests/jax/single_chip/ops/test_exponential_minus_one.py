# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from infra import ComparisonConfig, run_op_test_with_random_inputs
from utils import Category


@pytest.fixture
def comparison_config() -> ComparisonConfig:
    # Needs to have a bigger atol due to inaccuracies in the exp op on tt-metal.
    # These values are handtuned to the smallest possible until test passed.
    # See issue https://github.com/tenstorrent/tt-mlir/issues/1199.
    config = ComparisonConfig()
    config.atol.required_atol = 0.2
    config.allclose.atol = 0.07
    config.allclose.rtol = 0.06
    return config


# @pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.numpy.expm1",
    shlo_op_name="stablehlo.exponential_minus_one",
)
@pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)], ids=lambda val: f"{val}")
def test_exponential_minus_one(x_shape: tuple, comparison_config: ComparisonConfig):
    def expm1(x: jax.Array) -> jax.Array:
        return jnp.expm1(x)

    run_op_test_with_random_inputs(
        expm1, [x_shape], comparison_config=comparison_config
    )
