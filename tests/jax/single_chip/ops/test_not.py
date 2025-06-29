# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from infra import random_tensor, run_op_test
from utils import Category, convert_output_to_bfloat16


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.numpy.logical_not",
    shlo_op_name="stablehlo.not{LOGICAL}",
)
@pytest.mark.parametrize(
    ["shape"],
    [
        [(32, 32)],
        [(64, 64)],
    ],
    ids=lambda val: f"{val}",
)
def test_logical_not(shape: tuple):
    @convert_output_to_bfloat16
    def logical_not(a: jax.Array) -> jax.Array:
        return jnp.logical_not(a)

    input = random_tensor(shape, jnp.int32, minval=0, maxval=2, random_seed=3)
    run_op_test(logical_not, [input])


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.numpy.bitwise_not",
    shlo_op_name="stablehlo.not{BITWISE}",
)
@pytest.mark.parametrize(
    ["shape"],
    [
        [(32, 32)],
        [(64, 64)],
    ],
    ids=lambda val: f"{val}",
)
def test_bitwise_not(shape: tuple):
    def bitwise_not(a: jax.Array) -> jax.Array:
        return jnp.bitwise_not(a)

    input = random_tensor(shape, jnp.int32, minval=0, maxval=10, random_seed=3)
    run_op_test(bitwise_not, [input])
