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
def test_maximum(x_shape: tuple, y_shape: tuple, record_tt_xla_property: Callable):
    def maximum(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.maximum(x, y)

    record_binary_op_test_properties(
        record_tt_xla_property,
        "jax.numpy.maximum",
        "stablehlo.maximum",
    )

    run_op_test_with_random_inputs(maximum, [x_shape, y_shape])
