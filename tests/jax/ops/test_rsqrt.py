# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import jax
import jax.lax as jlx
import pytest
from infra import run_op_test_with_random_inputs
from utils import record_unary_op_test_properties


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)], ids=lambda val: f"{val}")
def test_rsqrt(x_shape: tuple, record_tt_xla_property: Callable):
    def rsqrt(x: jax.Array) -> jax.Array:
        return jlx.rsqrt(x)

    record_unary_op_test_properties(
        record_tt_xla_property,
        "jax.lax.rsqrt",
        "stablehlo.rsqrt",
    )

    # Input must be strictly positive because of sqrt(x).
    run_op_test_with_random_inputs(rsqrt, [x_shape], minval=0.1, maxval=10.0)
