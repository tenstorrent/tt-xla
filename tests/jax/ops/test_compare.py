# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import jax
import jax.lax as jlx
import jax.numpy as jnp
import pytest
from infra import run_op_test_with_random_inputs

# NOTE TTNN does not support boolean data type, so bfloat16 is used instead.
# Hence the output of comparison operation is bfloat16. JAX can not perform any
# computation due to mismatch in output data type (in testing infrastructure).
# The following tests explicitly convert data type of comparison operation
# output for the verification purposes.

# TODO Remove this workaround once the data type issue is resolved.
# https://github.com/tenstorrent/tt-xla/issues/93

# TODO investigate why this decorator cannot be removed. See issue
# https://github.com/tenstorrent/tt-xla/issues/156


def convert_output_to_bfloat16(f: Callable):
    """Decorator to work around the mentioned issue."""

    def wrapper(*args, **kwargs):
        res = f(*args, **kwargs)
        return jlx.convert_element_type(res, jnp.bfloat16)

    return wrapper


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_properties(
    test_category="op_test",
    jax_op_name="jax.numpy.equal",
    shlo_op_name="stablehlo.compare{EQ}",
)
@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [
        [(32, 32), (32, 32)],
        [(64, 64), (64, 64)],
    ],
    ids=lambda val: f"{val}",
)
def test_compare_equal(x_shape: tuple, y_shape: tuple):
    @convert_output_to_bfloat16
    def equal(x: jax.Array, y: jax.Array) -> jax.Array:
        return x == y

    run_op_test_with_random_inputs(equal, [x_shape, y_shape])


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_properties(
    test_category="op_test",
    jax_op_name="jax.numpy.not_equal",
    shlo_op_name="stablehlo.compare{NE}",
)
@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [
        [(32, 32), (32, 32)],
        [(64, 64), (64, 64)],
    ],
    ids=lambda val: f"{val}",
)
def test_compare_not_equal(x_shape: tuple, y_shape: tuple):
    @convert_output_to_bfloat16
    def not_equal(x: jax.Array, y: jax.Array) -> jax.Array:
        return x != y

    run_op_test_with_random_inputs(not_equal, [x_shape, y_shape])


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_properties(
    test_category="op_test",
    jax_op_name="jax.numpy.greater",
    shlo_op_name="stablehlo.compare{GT}",
)
@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [
        [(32, 32), (32, 32)],
        [(64, 64), (64, 64)],
    ],
    ids=lambda val: f"{val}",
)
def test_compare_greater(x_shape: tuple, y_shape: tuple):
    @convert_output_to_bfloat16
    def greater(x: jax.Array, y: jax.Array) -> jax.Array:
        return x > y

    run_op_test_with_random_inputs(greater, [x_shape, y_shape])


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_properties(
    test_category="op_test",
    jax_op_name="jax.numpy.greater_equal",
    shlo_op_name="stablehlo.compare{GE}",
)
@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [
        [(32, 32), (32, 32)],
        [(64, 64), (64, 64)],
    ],
    ids=lambda val: f"{val}",
)
def test_compare_greater_equal(x_shape: tuple, y_shape: tuple):
    @convert_output_to_bfloat16
    def greater_equal(x: jax.Array, y: jax.Array) -> jax.Array:
        return x >= y

    run_op_test_with_random_inputs(greater_equal, [x_shape, y_shape])


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_properties(
    test_category="op_test",
    jax_op_name="jax.numpy.less",
    shlo_op_name="stablehlo.compare{LT}",
)
@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [
        [(32, 32), (32, 32)],
        [(64, 64), (64, 64)],
    ],
    ids=lambda val: f"{val}",
)
def test_compare_less(x_shape: tuple, y_shape: tuple):
    @convert_output_to_bfloat16
    def less(x: jax.Array, y: jax.Array) -> jax.Array:
        return x < y

    run_op_test_with_random_inputs(less, [x_shape, y_shape])


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_properties(
    test_category="op_test",
    jax_op_name="jax.numpy.less_equal",
    shlo_op_name="stablehlo.compare{LE}",
)
@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [
        [(32, 32), (32, 32)],
        [(64, 64), (64, 64)],
    ],
    ids=lambda val: f"{val}",
)
def test_compare_less_equal(x_shape: tuple, y_shape: tuple):
    @convert_output_to_bfloat16
    def less_equal(x: jax.Array, y: jax.Array) -> jax.Array:
        return x <= y

    run_op_test_with_random_inputs(less_equal, [x_shape, y_shape])
