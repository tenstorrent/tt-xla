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


# @pytest.mark.push
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
            marks=pytest.mark.skip(
                reason="Unsupported data type which is not handled/casted by runtime"
            ),
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
        This test will push a scalar of a certain data type to device, and
        pull it back to host. If the data type is not supported, the runtime will cast
        it to a supported data type alias. When the host pulls the tensor back, the
        runtime will see that the data type we are expecting is different from the
        true data type of the runtime tensor, and will cast it to the type that the
        host is requesting.

        Scalars are actually 0-dim arrays. They can be created the same way arrays are,
        using `jax.array(<some-value>, dtype)` or using `dtype(<some-value>)`.
        """
        return inp

    # Pass in a jax array with the desired dtype to the program
    # This ensures that we will push a tensor to device, run a
    # program that does nothing, and pull the output back to host.
    with enable_x64():
        run_op_test(scalar, [jnp.array(1, dtype)])
