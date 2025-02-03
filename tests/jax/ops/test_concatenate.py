# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import jax
import jax.numpy as jnp
import pytest
from infra import run_op_test_with_random_inputs
from utils import record_binary_op_test_properties


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
def test_concatenate(
    x_shape: tuple, y_shape: tuple, axis: int, record_tt_xla_property: Callable
):
    def concat(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.concatenate([x, y], axis=axis)

    record_binary_op_test_properties(
        record_tt_xla_property,
        "jax.numpy.concatenate",
        "stablehlo.concatenate",
    )

    run_op_test_with_random_inputs(concat, [x_shape, y_shape])
