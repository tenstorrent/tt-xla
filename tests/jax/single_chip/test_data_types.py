# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
This file contains sanity tests which create arrays of various dtypes, in order not to
parametrize each test additionally with dtypes.
"""

import jax
import jax.numpy as jnp
import pytest
from infra import run_op_test
from jax._src.typing import DTypeLike
from utils import Category, enable_x64


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.OTHER)
@pytest.mark.parametrize(
    "dtype",
    [
        # uints
        jnp.uint8,
        jnp.uint16,
        jnp.uint32,
        jnp.uint64,
        # ints
        jnp.int8,
        jnp.int16,
        jnp.int32,
        jnp.int64,
        # floats
        pytest.param(
            jnp.float16,
            marks=pytest.mark.skip(reason="Unsupported data type"),
        ),
        jnp.float32,
        jnp.float64,
        # bfloat
        jnp.bfloat16,
        # bool
        jnp.bool,
    ],
)
def test_dtypes(dtype: DTypeLike):
    def scalar(inp) -> jax.Array:
        """
        This test just returns a scalar of a certain dtype. It will fail if dtype is
        unsupported. In mlir graph, it produces one simple stablehlo.constant op.

        Scalars are actually 0-dim arrays. They can be created the same way arrays are,
        using `jax.array(<some-value>, dtype)` or using `dtype(<some-value>)`.
        """
        return inp  # same as dtype(1)

    # Pass in a jax array with the desired dtype to the program
    # This ensures the we will push a tensor to device, run a
    # program tha does nothing, and pulls the output back to host
    with enable_x64():
        run_op_test(scalar, [jnp.array(1, dtype)])
