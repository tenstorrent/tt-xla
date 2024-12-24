# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from infra import run_op_test_with_random_inputs


@pytest.mark.parametrize("x_shape", [(32, 32), (64, 64)])
def test_log1p(x_shape: tuple):
    def log1p(x: jax.Array) -> jax.Array:
        return jnp.log1p(x)

    run_op_test_with_random_inputs(log1p, [x_shape])
