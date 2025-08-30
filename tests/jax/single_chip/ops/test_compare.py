# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import pytest
from infra import run_op_test_with_random_inputs
from utils import Category, convert_output_to_bfloat16


# @pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
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


# @pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
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


# @pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
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


# @pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
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


# @pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
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


# @pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
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
