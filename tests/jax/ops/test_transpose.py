# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import jax
import jax.numpy as jnp
import pytest
from infra import run_op_test_with_random_inputs
from utils import record_unary_op_test_properties


@pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)], ids=lambda val: f"{val}")
def test_transpose(x_shape: tuple, record_tt_xla_property: Callable):
    def transpose(x: jax.Array) -> jax.Array:
        return jnp.transpose(x)

    record_unary_op_test_properties(
        record_tt_xla_property,
        "jax.numpy.transpose",
        "stablehlo.transpose",
    )

    run_op_test_with_random_inputs(transpose, [x_shape])
