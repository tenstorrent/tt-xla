# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from infra import run_op_test_with_random_inputs
from utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.numpy.negative",
    shlo_op_name="stablehlo.negative",
)
@pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)], ids=lambda val: f"{val}")
@pytest.mark.parametrize("format", ["float32", "bfloat16"])
def test_negate(x_shape: tuple, format: str):
    def negate(x: jax.Array) -> jax.Array:
        return jnp.negative(x)

    if format == "float32":
        dtype = None
    else:
        dtype = jnp.bfloat16

    # Trying both negative and positive values.
    run_op_test_with_random_inputs(
        negate,
        [x_shape],
        minval=-5.0,
        maxval=5.0,
        dtype=dtype,
    )
