# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from infra import run_op_test_with_random_inputs
from utils import Category

# NOTE: This test passes `request` to support serialization (--serialize).
# Other op tests can follow this pattern. See docs/src/test_infra.md for details.


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.numpy.add",
    shlo_op_name="stablehlo.add",
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
def test_add(x_shape: tuple, y_shape: tuple, format: str, request):
    def add(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.add(x, y)

    if format == "float32":
        dtype = None
    else:
        dtype = jnp.bfloat16

    run_op_test_with_random_inputs(
        add,
        [x_shape, y_shape],
        dtype=dtype,
        request=request,
    )
