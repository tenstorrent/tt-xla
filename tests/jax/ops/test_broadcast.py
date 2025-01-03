# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import pytest
from infra import run_op_test_with_random_inputs

# TODO this skip reason is from old infra. Currently it fails with
# `error: type of return operand 0 ('tensor<2x1xf32>') doesn't match function result type ('tensor<2x4xf32>') in function @main`.
# Check why this happens and what is the state of this test in old infra (skip reason is old).


@pytest.mark.parametrize("input_shapes", [[(2, 1)]])
@pytest.mark.skip(
    "Broadcasted values are incorrect. "
    "Fails with: AssertionError: PCC is 0.37796446681022644 which is less than 0.99"
)
def test_broadcast_op(input_shapes):
    def module_broadcast(a):
        return jnp.broadcast_to(a, (2, 4))

    run_op_test_with_random_inputs(module_broadcast, input_shapes)
