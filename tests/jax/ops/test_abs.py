# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from infra import run_op_test_with_random_inputs


@pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)])
def test_abs(x_shape: tuple):
    def abs(x: jax.Array) -> jax.Array:
        return jnp.abs(x)

    # Test both negative and positive values.
    run_op_test_with_random_inputs(abs, [x_shape], minval=-5.0, maxval=5.0)