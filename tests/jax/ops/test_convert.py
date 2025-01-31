# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.lax as jlx
import jax.numpy as jnp
import pytest
from infra import random_tensor, run_op_test
from jax._src.typing import DTypeLike

# Allow 64bit precision in jax which is disabled by default.
jax.config.update("jax_enable_x64", True)

# NOTE Use test_data_types.py as reference for all supported data types.


@pytest.mark.parametrize(
    "from_dtype",
    [
        # uints
        pytest.param(
            jnp.uint8,
            marks=pytest.mark.skip(reason="Unsupported data type"),
        ),
        jnp.uint16,
        jnp.uint32,
        jnp.uint64,
        # ints
        pytest.param(
            jnp.int8,
            marks=pytest.mark.skip(reason="Unsupported data type"),
        ),
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
        pytest.param(
            jnp.bool,
            marks=pytest.mark.skip(
                reason="Cannot make random tensor of bools in current infra"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "to_dtype",
    [
        # uints
        pytest.param(
            jnp.uint8,
            marks=pytest.mark.skip(reason="Unsupported data type"),
        ),
        jnp.uint16,
        jnp.uint32,
        jnp.uint64,
        # ints
        pytest.param(
            jnp.int8,
            marks=pytest.mark.skip(reason="Unsupported data type"),
        ),
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
        pytest.param(
            jnp.bool,
            marks=pytest.mark.skip(
                reason="Cannot make random tensor of bools in current infra"
            ),
        ),
    ],
)
@pytest.mark.skip(reason="https://github.com/tenstorrent/tt-xla/issues/206")
def test_convert(from_dtype: DTypeLike, to_dtype: DTypeLike):
    def convert(x: jax.Array) -> jax.Array:
        return jlx.convert_element_type(x, new_dtype=to_dtype)

    x_shape = (32, 32)  # Shape does not make any impact here, thus not parametrized.
    input = random_tensor(x_shape, from_dtype, minval=0.0, maxval=10.0)

    run_op_test(convert, [input])
