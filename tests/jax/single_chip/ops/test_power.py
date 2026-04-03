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
    jax_op_name="jax.numpy.power",
    shlo_op_name="stablehlo.power",
)
@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [
        [(32, 32), (32, 32)],
        [(64, 64), (64, 64)],
    ],
    ids=lambda val: f"{val}",
)
@pytest.mark.parametrize("format", ["float32", "bfloat16"])
def test_power(x_shape: tuple, y_shape: tuple, format: str):
    def power(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.power(x, y)

    if format == "float32":
        dtype = None
    else:
        dtype = jnp.bfloat16

    run_op_test_with_random_inputs(power, [x_shape, y_shape], dtype=dtype)
