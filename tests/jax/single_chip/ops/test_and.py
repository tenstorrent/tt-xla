# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from infra import random_tensor, run_op_test
from utils import Category, convert_output_to_bfloat16


# @pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.numpy.logical_and",
    shlo_op_name="stablehlo.and{LOGICAL}",
)
@pytest.mark.parametrize(
    ["shape"],
    [
        [(32, 32)],
        [(64, 64)],
    ],
    ids=lambda val: f"{val}",
)
def test_logical_and(shape: tuple):
    @convert_output_to_bfloat16
    def logical_and(a: jax.Array, b: jax.Array) -> jax.Array:
        return jnp.logical_and(a, b)

    # TODO change int32 to bool https://github.com/tenstorrent/tt-xla/issues/339
    lhs = random_tensor(shape, jnp.int32, minval=0, maxval=2, random_seed=3)
    rhs = random_tensor(shape, jnp.int32, minval=0, maxval=2, random_seed=6)
    run_op_test(logical_and, [lhs, rhs])


# @pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.numpy.bitwise_and",
    shlo_op_name="stablehlo.and{BITWISE}",
)
@pytest.mark.parametrize(
    ["shape"],
    [
        [(32, 32)],
        [(64, 64)],
    ],
    ids=lambda val: f"{val}",
)
def test_bitwise_and(shape: tuple):
    def bitwise_and(a: jax.Array, b: jax.Array) -> jax.Array:
        return jnp.bitwise_and(a, b)

    lhs = random_tensor(shape, jnp.int32, minval=0, maxval=10, random_seed=3)
    rhs = random_tensor(shape, jnp.int32, minval=0, maxval=10, random_seed=6)
    run_op_test(bitwise_and, [lhs, rhs])
