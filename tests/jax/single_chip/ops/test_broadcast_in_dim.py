# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from infra import run_op_test_with_random_inputs

from tests.utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    jax_op_name="jax.numpy.broadcast_to",
    shlo_op_name="stablehlo.broadcast_in_dim",
)
@pytest.mark.parametrize("input_shapes", [[(2, 1)]], ids=lambda val: f"{val}")
def test_broadcast_in_dim(input_shapes: tuple):
    def broadcast(a: jax.Array):
        return jnp.broadcast_to(a, (2, 4))

    run_op_test_with_random_inputs(broadcast, input_shapes)
