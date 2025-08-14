# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
This file contains sanity tests if scalars work as expected, since they represent a
special case of 0-dim arrays.
"""

import jax
import pytest
from infra import run_op_test
from jax import numpy as jnp
from utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OTHER)
def test_scalar_scalar_add():
    """Tests adding two scalars."""

    def add() -> jax.Array:
        return jnp.array(1, jnp.float32) + jnp.array(2, jnp.float32)

    run_op_test(add, [])


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OTHER)
def test_scalar_array_add():
    """
    Tests adding scalar and an array.

    Also performs a sanity check that addition is commutative, that is
    scalar + array is the same as array + scalar.
    """

    def array_plus_scalar() -> jax.Array:
        return jnp.ones((32, 32), jnp.float32) + jnp.array(2.0, jnp.float32)

    def scalar_plus_array() -> jax.Array:
        return jnp.array(2.0, jnp.float32) + jnp.ones((32, 32), jnp.float32)

    run_op_test(array_plus_scalar, [])
    run_op_test(scalar_plus_array, [])
