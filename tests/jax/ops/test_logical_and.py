# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable
import jax
import jax.numpy as jnp
import pytest
from infra import random_tensor, run_op_test
from utils import convert_output_to_bfloat16, record_binary_op_test_properties


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.parametrize(
    ["shape"],
    [
        [(32, 32)],
        [(64, 64)],
    ],
    ids=lambda val: f"{val}",
)
def test_logical_and(shape: tuple, record_tt_xla_property: Callable):
    @convert_output_to_bfloat16
    def logical_and(a: jax.Array, b: jax.Array) -> jax.Array:
        return jnp.logical_and(a, b)

    record_binary_op_test_properties(
        record_tt_xla_property,
        "jax.numpy.logical_and",
        "stablehlo.and",
    )

    lhs = random_tensor(shape, jnp.int32, minval=0, maxval=2, random_seed=3)
    rhs = random_tensor(shape, jnp.int32, minval=0, maxval=2, random_seed=6)
    run_op_test(logical_and, [lhs, rhs])
