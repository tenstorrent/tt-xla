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
    jax_op_name="jax.numpy.concatenate",
    shlo_op_name="stablehlo.concatenate",
)
@pytest.mark.parametrize(
    ["x_shape", "y_shape", "axis"],
    [
        [(32, 32), (32, 32), 0],
        [(64, 64), (64, 64), 1],
        [(32, 32, 32), (32, 32, 32), 2],
        [(64, 64, 64, 64), (64, 64, 64, 64), 3],
    ],
    ids=lambda val: f"{val}",
)
def test_concatenate(x_shape: tuple, y_shape: tuple, axis: int):
    def concat(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.concatenate([x, y], axis=axis)

    run_op_test_with_random_inputs(concat, [x_shape, y_shape])
