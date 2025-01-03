# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import pytest
from infra import run_op_test_with_random_inputs


def add(x: jax.Array, y: jax.Array) -> jax.Array:
    return jax.numpy.add(x, y)


@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [
        [(32, 32), (32, 32)],
        [(64, 64), (64, 64)],
    ],
)
def test_add(x_shape: tuple, y_shape: tuple):
    run_op_test_with_random_inputs(add, [x_shape, y_shape])
