# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Union

import jax
import jax.numpy as jnp
import pytest
from infra import run_op_test, run_op_test_with_random_inputs


@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [
        [(32, 32), (32, 32)],
        [(64, 64), (64, 64)],
    ],
)
def test_add(x_shape: tuple, y_shape: tuple):
    def add(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.add(x, y)

    run_op_test_with_random_inputs(add, [x_shape, y_shape])


# Convenience alias.
scalar_or_array = Union[int, float, jax.Array]


# TODO should we test for other types like bfloat16 and uint16?
@pytest.mark.parametrize(
    ["in0", "in1"],
    [
        # It is assumed that add is commutative.
        [jnp.array(1.0), jnp.float32(2.0)],  # Float array + float scalar
        [jnp.array(1.0), jnp.uint32(2)],  # Float array + int scalar
        [jnp.uint32(1), jnp.float32(2.0)],  # Int scalar + float scalar
        [jnp.float32(1.0), jnp.float32(2.0)],  # Float scalar + float scalar
        [jnp.array(1, dtype=jnp.uint32), jnp.float32(2.0)],  # Int array + float scalar
        pytest.param(
            *[jnp.uint32(1), jnp.uint32(2)],  # Int scalar + int scalar
            marks=pytest.mark.skip(
                reason="Atol comparison failed. Calculated: atol=20971516.0"
            ),
        ),
        pytest.param(
            *[jnp.array(1, dtype=jnp.uint32), jnp.uint32(2)],  # Int array + int scalar
            marks=pytest.mark.skip(
                reason="Atol comparison failed. Calculated: atol=20971516.0"
            ),
        ),
    ],
)
def test_scalar_add(in0: scalar_or_array, in1: scalar_or_array):
    def add(x: scalar_or_array, y: scalar_or_array) -> scalar_or_array:
        return x + y

    run_op_test(add, [in0, in1])
