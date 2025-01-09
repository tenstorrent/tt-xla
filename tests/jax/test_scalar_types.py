# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Union

import jax
import jax.numpy as jnp
import pytest
from infra import run_op_test, supported_dtypes
from jax._src.typing import DTypeLike

# Convenience alias.
scalar = Union[int, float]

# Convenience alias.
scalar_or_array = Union[scalar, jax.Array]


@pytest.mark.parametrize(
    ["dtype0", "dtype1"],
    [(d0, d1) for d0 in supported_dtypes for d1 in supported_dtypes],
)
@pytest.mark.skip(
    "Passes locally but fails on CI due to AssertionError: Unexpected XLA layout override"
)
def test_scalar_scalar_add(dtype0: DTypeLike, dtype1: DTypeLike):
    """
    Tests adding of two scalars.

    Adding two ints causes huge atol differences. It is known that tt-metal does not
    work well with ints. Adding an int and a float works due to upcast to float.
    """

    def add(x: scalar, y: scalar) -> scalar:
        return x + y

    in0, in1 = dtype0(1), dtype1(2)

    if in0.dtype in [jnp.uint32, jnp.uint16] and in1.dtype in [jnp.uint32, jnp.uint16]:
        pytest.skip("Adding two ints causes huge atol differences.")

    run_op_test(add, [in0, in1])


@pytest.mark.parametrize(
    ["in0", "in1"],
    [
        # Scalar and 0-dim array.
        [jnp.array(1.0, jnp.float32), jnp.float32(2.0)],
        [jnp.float32(2.0), jnp.array(1.0, jnp.float32)],
        # Scalar and 1-dim array.
        [jnp.ones((32,), jnp.float32), jnp.float32(2.0)],
        [jnp.float32(2.0), jnp.ones((32,), jnp.float32)],
        # Scalar and 2-dim array.
        [jnp.ones((1, 32), jnp.float32), jnp.float32(2.0)],
        [jnp.float32(2.0), jnp.ones((1, 32), jnp.float32)],
    ],
)
@pytest.mark.skip(
    "Passes locally but fails on CI due to AssertionError: Unexpected XLA layout override"
)
def test_scalar_array_add(in0: scalar_or_array, in1: scalar_or_array):
    """
    Tests adding of scalar and an array.

    Also performs a sanity check that addition is commutative, that is
    scalar + array is the same as array + scalar.
    """

    def add(x: scalar_or_array, y: scalar_or_array) -> scalar_or_array:
        return x + y

    run_op_test(add, [in0, in1])
