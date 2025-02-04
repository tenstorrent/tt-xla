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


# TODO we need to parametrize with all supported dtypes.
@pytest.mark.parametrize(
    "from_dtype",
    [
        "bfloat16",
        "float32",
    ],
)
@pytest.mark.parametrize(
    "to_dtype",
    [
        "uint32",
        "uint64",
        "int32",
        "int64",
        "bfloat16",
        "float32",
        "float64",
    ],
)
@pytest.mark.skip(
    f"Skipped unconditionally due to many fails. There is ongoing work on rewriting these tests."
)
def test_convert(
    from_dtype: DTypeLike, to_dtype: DTypeLike, record_tt_xla_property: Callable
):
    def convert(x: jax.Array) -> jax.Array:
        return jlx.convert_element_type(x, new_dtype=jnp.dtype(to_dtype))

    record_unary_op_test_properties(
        record_tt_xla_property,
        "jax.lax.convert_element_type",
        "stablehlo.convert",
    )

    x_shape = (32, 32)  # Shape does not make any impact here, thus not parametrized.
    input = random_tensor(x_shape, dtype=from_dtype)

    run_op_test(convert, [input])
