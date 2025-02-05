# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import jax
import jax.numpy as jnp
import pytest
from infra import run_op_test_with_random_inputs
from utils import record_unary_op_test_properties


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)], ids=lambda val: f"{val}")
def test_exponential(x_shape: tuple, record_tt_xla_property: Callable):
    def exponential(x: jax.Array) -> jax.Array:
        return jnp.exp(x)

    record_unary_op_test_properties(
        record_tt_xla_property,
        "jax.numpy.exp",
        "stablehlo.exponential",
    )

    run_op_test_with_random_inputs(exponential, [x_shape])
