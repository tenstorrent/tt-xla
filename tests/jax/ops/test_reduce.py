# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from infra import ComparisonConfig, run_op_test_with_random_inputs


# TODO investigate why this doesn't pass with default comparison config.
@pytest.fixture
def comparison_config() -> ComparisonConfig:
    config = ComparisonConfig()
    config.disable_all()
    config.pcc.enable()
    config.pcc.required_pcc = 0.95
    return config


# TODO axis should be parametrized as well.
@pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)])
def test_reduce_sum(x_shape: tuple, comparison_config: ComparisonConfig):
    def reduce_sum(x: jax.Array) -> jax.Array:
        return jnp.sum(x)

    run_op_test_with_random_inputs(
        reduce_sum, [x_shape], comparison_config=comparison_config
    )


# TODO axis should be parametrized as well.
@pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)])
def test_reduce_max(x_shape: tuple, comparison_config: ComparisonConfig):
    def reduce_max(x: jax.Array) -> jax.Array:
        return jnp.max(x)

    run_op_test_with_random_inputs(
        reduce_max, [x_shape], comparison_config=comparison_config
    )


# TODO add tests for reduce `and` and reduce `or`.
