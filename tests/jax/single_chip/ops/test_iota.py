# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import jax.lax as lax
import pytest
from infra import run_op_test_with_random_inputs

from tests.utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.lax.iota",
    shlo_op_name="stablehlo.iota",
)
@pytest.mark.parametrize("length", [32, 64], ids=lambda val: f"length={val}")
def test_iota(length: int):
    def iota(_: jax.Array) -> jax.Array:
        return lax.iota(jnp.float32, length)

    run_op_test_with_random_inputs(iota, [(1,)])
