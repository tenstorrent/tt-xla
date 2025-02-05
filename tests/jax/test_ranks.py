# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
This file contains sanity tests for some representative ops to make sure they work for
various ranks, in order not to parametrize each test additionally with ranks.
"""

import jax
import pytest
from infra import run_op_test_with_random_inputs
from jax import numpy as jnp


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.parametrize(
    "x_shape",
    [
        pytest.param(
            (),
            marks=pytest.mark.skip(
                reason=(
                    "Unexpected XLA layout override. "
                    "See issue https://github.com/tenstorrent/tt-xla/issues/173"
                )
            ),
        ),
        (32,),
        (32, 32),
        (1, 32, 32),
        (1, 3, 32, 32),
    ],
)
def test_unary_op(x_shape: tuple):
    """Using negative as it is trivial, since this test only focuses on ranks."""

    def negate(x: jax.Array) -> jax.Array:
        return jnp.negative(x)

    run_op_test_with_random_inputs(negate, [x_shape])


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(
            (),
            marks=pytest.mark.skip(
                reason=(
                    "Unexpected XLA layout override. "
                    "See issue https://github.com/tenstorrent/tt-xla/issues/173"
                )
            ),
        ),
        (32,),
        (32, 32),
        (1, 32, 32),
        (1, 3, 32, 32),
    ],
)
def test_binary_op(shape: tuple):
    """Using add as it is trivial, since this test only focuses on ranks."""

    def add(x: jax.Array, y: jax.Array) -> jax.Array:
        return x + y

    run_op_test_with_random_inputs(add, [shape, shape])
