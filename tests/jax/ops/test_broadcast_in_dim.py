# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import pytest
from infra import run_op_test_with_random_inputs


@pytest.mark.parametrize("input_shapes", [[(2, 1)]])
@pytest.mark.xfail(
    reason="AssertionError: Atol comparison failed. Calculated: atol=0.804124116897583. Required: atol=0.16"
)
def test_broadcast_in_dim(input_shapes):
    def broadcast(a):
        return jnp.broadcast_to(a, (2, 4))

    run_op_test_with_random_inputs(broadcast, input_shapes)
