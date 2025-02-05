# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import jax
import jax.numpy as jnp
import pytest
from infra import run_op_test_with_random_inputs
from utils import record_binary_op_test_properties


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [
        [(32, 32), (32, 32)],
        [(64, 64), (64, 64)],
    ],
    ids=lambda val: f"{val}",
)
def test_divide(x_shape: tuple, y_shape: tuple, record_tt_xla_property: Callable):
    def divide(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.divide(x, y)

    record_binary_op_test_properties(
        record_tt_xla_property, "jax.numpy.divide", "stablehlo.divide"
    )

    run_op_test_with_random_inputs(divide, [x_shape, y_shape])
