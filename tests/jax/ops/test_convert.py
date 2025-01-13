# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import jax
import jax.lax as jlx
import jax.numpy as jnp
import pytest
from infra import random_tensor, run_op_test
from jax._src.typing import DTypeLike
from utils import record_unary_op_test_properties

# Allow 64bit precision in jax which is disabled by default.
jax.config.update("jax_enable_x64", True)

# NOTE Use test_data_types.py as reference for all supported data types.


@pytest.mark.parametrize(
    "from_dtype",
    [
        jnp.uint16,
        jnp.uint32,
        pytest.param(
            jnp.uint64,
            marks=pytest.mark.skip(
                reason=(
                    "Cannot get the device from a tensor with host storage. "
                    "See issue https://github.com/tenstorrent/tt-xla/issues/171"
                )
            ),
        ),
        pytest.param(
            jnp.int16,
            marks=pytest.mark.skip(
                reason=(
                    "Cannot get the device from a tensor with host storage. "
                    "See issue https://github.com/tenstorrent/tt-xla/issues/171"
                )
            ),
        ),
        pytest.param(
            jnp.int32,
            marks=pytest.mark.skip(
                reason=(
                    "Cannot get the device from a tensor with host storage. "
                    "See issue https://github.com/tenstorrent/tt-xla/issues/171"
                )
            ),
        ),
        pytest.param(
            jnp.int64,
            marks=pytest.mark.skip(
                reason=(
                    "Cannot get the device from a tensor with host storage. "
                    "See issue https://github.com/tenstorrent/tt-xla/issues/171"
                )
            ),
        ),
        jnp.float32,
        jnp.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "to_dtype",
    [
        pytest.param(
            jnp.uint16,
            marks=pytest.mark.skip(
                reason=(
                    "Fails due to low comparison metrics. "
                    "See issue https://github.com/tenstorrent/tt-xla/issues/172"
                )
            ),
        ),
        jnp.uint32,
        jnp.uint64,
        pytest.param(
            jnp.int16,
            marks=pytest.mark.skip(
                reason=(
                    "Fails due to low comparison metrics. "
                    "See issue https://github.com/tenstorrent/tt-xla/issues/172"
                )
            ),
        ),
        jnp.int32,
        jnp.int64,
        jnp.float32,
        jnp.bfloat16,
    ],
)
@pytest.mark.skip(
    f"Skipped unconditionally due to many fails. There is ongoing work on rewriting these tests."
)
def test_convert(
    from_dtype: DTypeLike, to_dtype: DTypeLike, record_tt_xla_property: Callable
):
    def convert(x: jax.Array) -> jax.Array:
        return jlx.convert_element_type(x, new_dtype=to_dtype)

    record_unary_op_test_properties(
        record_tt_xla_property,
        "jax.lax.convert_element_type",
        "stablehlo.convert",
    )

    x_shape = (32, 32)  # Shape does not make any impact here, thus not parametrized.
    input = random_tensor(x_shape, from_dtype, minval=0.0, maxval=10.0)

    run_op_test(convert, [input])
