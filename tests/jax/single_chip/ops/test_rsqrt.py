# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.lax as jlx
import pytest
from infra import run_op_test_with_random_inputs
from utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.lax.rsqrt",
    shlo_op_name="stablehlo.rsqrt",
)
@pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)], ids=lambda val: f"{val}")
def test_rsqrt(x_shape: tuple):
    def rsqrt(x: jax.Array) -> jax.Array:
        return jlx.rsqrt(x)

    # Input must be strictly positive because of sqrt(x).
    run_op_test_with_random_inputs(rsqrt, [x_shape], minval=0.1, maxval=10.0)
