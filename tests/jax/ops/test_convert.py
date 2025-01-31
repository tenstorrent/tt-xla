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
from utils import compile_fail, record_unary_op_test_properties, runtime_fail

from tests.utils import enable_x64

# NOTE Use test_data_types.py as reference for all supported data types.


def conditionally_skip(from_dtype: DTypeLike, to_dtype: DTypeLike):
    """
    Helper function which checks dtype combination and skips if unsupported for some
    reason.

    Extracted here in order not to pollute the test function.
    """
    # ---------- Atol comparison failed ----------

    if from_dtype == jnp.uint32 and to_dtype in [jnp.uint16, jnp.int16]:
        pytest.xfail(
            runtime_fail(
                "AssertionError: Atol comparison failed. Calculated: atol=9.0. Required: atol=0.16."
            )
        )

    if from_dtype == jnp.float64 and to_dtype == jnp.uint16:
        pytest.xfail(
            runtime_fail(
                "AssertionError: Atol comparison failed. Calculated: atol=9.0. Required: atol=0.16."
            )
        )

    if from_dtype == jnp.float32 and to_dtype in [jnp.uint16, jnp.int16]:
        pytest.xfail(
            runtime_fail(
                "AssertionError: Atol comparison failed. Calculated: atol=1.0. Required: atol=0.16."
            )
        )

    if from_dtype == jnp.bfloat16 and to_dtype in [jnp.uint16, jnp.int16]:
        pytest.xfail(
            runtime_fail(
                "AssertionError: Atol comparison failed. Calculated: atol=1.0. Required: atol=0.16."
            )
        )

    # ---------- Cannot get the device from a tensor with host storage ----------

    if from_dtype == jnp.uint64 and to_dtype in [
        jnp.uint16,
        jnp.uint32,
        jnp.uint64,
        jnp.int16,
        jnp.int32,
        jnp.int64,
    ]:
        pytest.xfail(
            runtime_fail(
                "Cannot get the device from a tensor with host storage "
                "(https://github.com/tenstorrent/tt-xla/issues/171)"
            )
        )

    if from_dtype in [jnp.int16, jnp.int32, jnp.int64] and to_dtype in [
        jnp.uint16,
        jnp.uint32,
        jnp.uint64,
        jnp.int16,
        jnp.int32,
        jnp.int64,
    ]:
        pytest.xfail(
            runtime_fail(
                "Cannot get the device from a tensor with host storage "
                "(https://github.com/tenstorrent/tt-xla/issues/171)"
            )
        )

    if from_dtype == jnp.float64 and to_dtype in [
        jnp.uint16,
        jnp.uint32,
        jnp.uint64,
        jnp.int16,
        jnp.int32,
        jnp.int64,
    ]:
        pytest.xfail(
            runtime_fail(
                "Cannot get the device from a tensor with host storage "
                "(https://github.com/tenstorrent/tt-xla/issues/171)"
            )
        )

    if to_dtype in [jnp.float32, jnp.float64] and from_dtype in [
        jnp.uint64,
        jnp.int16,
        jnp.int32,
        jnp.int64,
        jnp.float64,
    ]:
        pytest.xfail(
            runtime_fail(
                "Cannot get the device from a tensor with host storage "
                "(https://github.com/tenstorrent/tt-xla/issues/171)"
            )
        )

    if to_dtype == jnp.bfloat16 and from_dtype in [
        jnp.uint64,
        jnp.int16,
        jnp.int32,
        jnp.int64,
        jnp.float64,
    ]:
        pytest.xfail(
            runtime_fail(
                "Cannot get the device from a tensor with host storage "
                "(https://github.com/tenstorrent/tt-xla/issues/171)"
            )
        )

    # ---------- Executable expected parameter x of size y but got... ----------

    if (
        from_dtype in [jnp.uint16, jnp.uint32, jnp.float32, jnp.bfloat16]
        and to_dtype == jnp.float64
    ):
        pytest.xfail(
            compile_fail(
                "Executable expected parameter 0 of size 8192 but got buffer with "
                "incompatible size 4096 (https://github.com/tenstorrent/tt-xla/issues/170)"
            )
        )


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
                reason="Causes segfaults. Should be investigated separately."
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
                reason="Causes segfaults. Should be investigated separately."
            ),
        ),
    ],
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

    # Some dtype conversions are not supported. Check and decide whether to skip or
    # proceed.
    conditionally_skip(from_dtype, to_dtype)

    # Shape does not make any impact here, thus not parametrized.
    x_shape = (32, 32)

    with enable_x64():
        input = random_tensor(x_shape, from_dtype, minval=0.0, maxval=10.0)

        run_op_test(convert, [input])
