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


@pytest.mark.parametrize("dtype", supported_dtypes)
@pytest.mark.skip(
    "Passes locally but fails on CI due to AssertionError: Unexpected XLA layout override"
)
def test_scalar_dtype(dtype: DTypeLike):
    """
    This test just returns a scalar of a certain dtype. It will fail if dtype is
    unsupported.
    """

    def add(x: scalar) -> scalar:
        return x

    in0 = dtype(1)  # Dummy scalar used as input.
    run_op_test(add, [in0])


@pytest.mark.parametrize("dtype", supported_dtypes)
@pytest.mark.skip(
    "Passes locally but fails on CI due to AssertionError: Unexpected XLA layout override"
)
def test_array_dtype(dtype: DTypeLike):
    """
    This test just returns an array of a certain dtype. It will fail if dtype is
    unsupported.
    """

    def array(x: jax.Array) -> jax.Array:
        return x

    in0 = jnp.ones((32, 32), dtype)  # Dummy array used as input.
    run_op_test(array, [in0])
