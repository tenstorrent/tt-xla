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

# Allow 64bit precision in jax which is disabled by default.
jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(
            jnp.uint8,
            marks=pytest.mark.xfail(reason="Unsupported data type"),
        ),
        jnp.uint16,
        jnp.uint32,
        jnp.uint64,
        pytest.param(
            jnp.int8,
            marks=pytest.mark.xfail(reason="Unsupported data type"),
        ),
        jnp.int16,
        jnp.int32,
        jnp.int64,
        pytest.param(
            jnp.float16,
            marks=pytest.mark.xfail(reason="Unsupported data type"),
        ),
        jnp.float32,
        pytest.param(
            jnp.float64,
            marks=pytest.mark.skip(
                reason=(
                    "Executable expected parameter 0 of size 8 but got buffer "
                    "with incompatible size 4. See issue "
                    "https://github.com/tenstorrent/tt-xla/issues/170"
                )
            ),
        ),
        jnp.bfloat16,
    ],
)
def test_dtypes(dtype: DTypeLike):
    def scalar() -> jax.Array:
        """
        This test just returns a scalar of a certain dtype. It will fail if dtype is
        unsupported.

        Scalars are actually 0-dim arrays. They can be created the same way arrays are,
        using `jax.array(<some-value>, dtype)` or using `dtype(<some-value>)`.
        """
        return jnp.array(1, dtype)  # same as dtype(1)

    run_op_test(scalar, [])
