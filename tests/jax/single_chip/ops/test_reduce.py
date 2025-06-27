# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from infra import ComparisonConfig, run_op_test_with_random_inputs
from utils import Category, failed_runtime


# TODO investigate why this doesn't pass with default comparison config.
@pytest.fixture
def comparison_config() -> ComparisonConfig:
    config = ComparisonConfig()
    config.disable_all()
    config.pcc.enable()
    config.pcc.required_pcc = 0.95
    return config


# TODO axis should be parametrized as well.
@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.numpy.sum",
    shlo_op_name="stablehlo.reduce{SUM}",
)
@pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)], ids=lambda val: f"{val}")
def test_reduce_sum(x_shape: tuple, comparison_config: ComparisonConfig):
    def reduce_sum(x: jax.Array) -> jax.Array:
        return jnp.sum(x)

    run_op_test_with_random_inputs(
        reduce_sum, [x_shape], comparison_config=comparison_config
    )


# TODO axis should be parametrized as well.
@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.numpy.max",
    shlo_op_name="stablehlo.reduce{MAX}",
)
@pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)], ids=lambda val: f"{val}")
def test_reduce_max(x_shape: tuple, comparison_config: ComparisonConfig):
    def reduce_max(x: jax.Array) -> jax.Array:
        return jnp.max(x)

    run_op_test_with_random_inputs(
        reduce_max, [x_shape], comparison_config=comparison_config
    )


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.numpy.all",
    shlo_op_name="stablehlo.reduce{AND}",
)
@pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)], ids=lambda val: f"{val}")
@pytest.mark.xfail(
    reason=failed_runtime(
        "i1 (boolean) output type mismatch on TTNN backend"
        "https://github.com/tenstorrent/tt-xla/issues/668"
    )
)
def test_reduce_and(x_shape: tuple, comparison_config: ComparisonConfig):
    def reduce_and(x: jax.Array) -> jax.Array:
        x_bool = x > 0.5
        return jnp.all(x_bool)

    run_op_test_with_random_inputs(
        reduce_and, [x_shape], comparison_config=comparison_config
    )


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.numpy.any",
    shlo_op_name="stablehlo.reduce{OR}",
)
@pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)], ids=lambda val: f"{val}")
@pytest.mark.xfail(
    reason=failed_runtime(
        "i1 (boolean) output type mismatch on TTNN backend"
        "https://github.com/tenstorrent/tt-xla/issues/668"
    )
)
def test_reduce_or(x_shape: tuple, comparison_config: ComparisonConfig):
    def reduce_or(x: jax.Array) -> jax.Array:
        x_bool = x > 0.5
        return jnp.any(x_bool)

    run_op_test_with_random_inputs(
        reduce_or, [x_shape], comparison_config=comparison_config
    )


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.numpy.prod",
    shlo_op_name="stablehlo.reduce{MULTIPLY}",
)
@pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)], ids=lambda val: f"{val}")
@pytest.mark.xfail(
    reason=failed_runtime(
        "'ttnn.prod' op TTNN only supports Reduce(prod) along all dimensions for bfloat16 datatype"
        "https://github.com/tenstorrent/tt-xla/issues/669"
    )
)
def test_reduce_multiply(x_shape: tuple, comparison_config: ComparisonConfig):
    def reduce_multiply(x: jax.Array) -> jax.Array:
        return jnp.prod(x, axis=None)

    run_op_test_with_random_inputs(
        reduce_multiply, [x_shape], comparison_config=comparison_config
    )


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.numpy.min",
    shlo_op_name="stablehlo.reduce{MIN}",
)
@pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)], ids=lambda val: f"{val}")
def test_reduce_min(x_shape: tuple, comparison_config: ComparisonConfig):
    def reduce_min(x: jax.Array) -> jax.Array:
        return jnp.min(x)

    run_op_test_with_random_inputs(
        reduce_min, [x_shape], comparison_config=comparison_config
    )
