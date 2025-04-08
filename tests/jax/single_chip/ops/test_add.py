# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from infra import run_op_test, random_tensor


from tests.utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.numpy.add",
    shlo_op_name="stablehlo.add",
)
@pytest.mark.parametrize(
    ["x_shape", "y_shape"],
    [
        [(64, 64), (64, 64)],
    ],
    ids=lambda val: f"{val}",
)
def test_add(x_shape: tuple, y_shape: tuple):
    def add(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.add(x, y)

    a = random_tensor(x_shape, dtype="bfloat16")
    b = random_tensor(y_shape, dtype="bfloat16")
    run_op_test(add, [a, b])
