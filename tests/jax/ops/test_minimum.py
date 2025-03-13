# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from infra import run_op_test_with_random_inputs

from tests.utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.numpy.minimum",
    shlo_op_name="stablehlo.minimum",
)
@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [
        [(32, 32), (32, 32)],
        [(64, 64), (64, 64)],
    ],
    ids=lambda val: f"{val}",
)
def test_minimum(x_shape: tuple, y_shape: tuple):
    def minimum(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.minimum(x, y)

    run_op_test_with_random_inputs(minimum, [x_shape, y_shape])
