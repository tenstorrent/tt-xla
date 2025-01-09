# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from infra import run_op_test_with_random_inputs


@pytest.mark.parametrize(
    ["in_shape", "out_shape"],
    [
        ((8, 32, 256), (2, 4, 32, 256)),
        ((8, 32, 32), (1, 2, 4, 32, 32)),
        ((8192, 128), (1, 256, 32, 128)),
    ],
)
def test_reshape(in_shape: tuple, out_shape: tuple):
    def reshape(x: jax.Array):
        return jnp.reshape(x, out_shape)

    run_op_test_with_random_inputs(reshape, [in_shape])
