# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.lax as jlx
import pytest
from infra import run_op_test_with_random_inputs


@pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)])
def test_rsqrt(x_shape: tuple):
    def rsqrt(x: jax.Array) -> jax.Array:
        return jlx.rsqrt(x)

    # Input must be strictly positive because of sqrt(x).
    run_op_test_with_random_inputs(rsqrt, [x_shape], minval=0.1, maxval=10.0)
