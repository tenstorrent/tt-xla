# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import jax
import pytest
from infra import run_op_test_with_random_inputs
from utils import record_binary_op_test_properties


# Tests for dot_general op where vectors containing indices of contracting dimensions
# are of size 1 and are equal. In training models, besides cases that correspond to matmul,
# this is the most common one we have.
@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [
        [(1, 32), (1, 32)],
        [(1, 32, 64), (1, 32, 32)],
        [(2, 32, 64), (2, 32, 64)],
        [(2, 16, 32, 64), (2, 16, 64, 32)],
    ],
    ids=lambda val: f"{val}",
)
def test_dot_general_common(
    x_shape: tuple, y_shape: tuple, record_tt_xla_property: Callable
):
    def dot_general(x: jax.Array, y: jax.Array) -> jax.Array:
        return jax.lax.dot_general(x, y, dimension_numbers=((1, 1), (0, 0)))

    record_binary_op_test_properties(
        record_tt_xla_property, "jax.lax.dot_general", "stablehlo.dot_general"
    )

    run_op_test_with_random_inputs(dot_general, [x_shape, y_shape])


# Tests for dot_general op where this operation corresponds to regular matmul.
@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [
        [(1, 32, 64), (1, 64, 32)],
        [(2, 32, 64), (2, 64, 64)],
    ],
)
def test_dot_general_matmul(
    x_shape: tuple, y_shape: tuple, record_tt_xla_property: Callable
):
    def dot_general(x: jax.Array, y: jax.Array) -> jax.Array:
        return jax.lax.dot_general(x, y, dimension_numbers=((2, 1), (0, 0)))

    record_binary_op_test_properties(
        record_tt_xla_property, "jax.lax.dot_general", "stablehlo.dot_general"
    )

    run_op_test_with_random_inputs(dot_general, [x_shape, y_shape])


# Tests for dot_general op where vectors containing indices of
# contracting dimensions are of size greater than 1.
@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [
        [(1, 16, 16, 8), (1, 16, 8, 16)],
        [(2, 8, 8, 16), (2, 8, 16, 8)],
    ],
)
def test_dot_general_multiple_contract(
    x_shape: tuple, y_shape: tuple, record_tt_xla_property: Callable
):
    def dot_general(x: jax.Array, y: jax.Array) -> jax.Array:
        return jax.lax.dot_general(x, y, dimension_numbers=(((1, 3), (1, 2)), (0, 0)))

    record_binary_op_test_properties(
        record_tt_xla_property, "jax.lax.dot_general", "stablehlo.dot_general"
    )

    run_op_test_with_random_inputs(dot_general, [x_shape, y_shape])
