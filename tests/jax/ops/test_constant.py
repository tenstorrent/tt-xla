# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import pytest
from infra import run_op_test_with_random_inputs


# TODO this skip reason is from old infra. Currently it fails with
# `error: type of return operand 0 ('tensor<1xf32>') doesn't match function result type ('tensor<3x3xf32>') in function @main`.
# Check why this happens and what is the state of this test in old infra (skip reason is old).
@pytest.mark.parametrize("input_shapes", [[(3, 3)]])
@pytest.mark.skip("AssertionError: ATOL is 21574.4375 which is greater than 0.01")
def test_constant_op(input_shapes):
    def module_constant_zeros(a):
        zeros = jnp.zeros(a.shape)
        return zeros

    def module_constant_ones(a):
        ones = jnp.ones(a.shape)
        return ones

    run_op_test_with_random_inputs(module_constant_zeros, input_shapes)
    run_op_test_with_random_inputs(module_constant_ones, input_shapes)


# TODO this was copied from old infra. It does nothing with `a` argument. Figure out what
# was meant to be tested this way.
@pytest.mark.parametrize("input_shapes", [[(3, 3)]])
@pytest.mark.skip("Fails due to: error: failed to legalize operation 'ttir.constant'")
def test_constant_op_multi_dim(input_shapes):
    def module_constant_multi(a):
        multi = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)
        return multi

    run_op_test_with_random_inputs(module_constant_multi, input_shapes)
