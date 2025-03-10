# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax.numpy as jnp
import pytest
from infra import run_op_test
from utils import TestCategory, compile_fail


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    test_category=TestCategory.OP_TEST.value,
    jax_op_name="jax.numpy.zeros",
    shlo_op_name="stablehlo.constant",
)
@pytest.mark.parametrize("shape", [(32, 32), (1, 1)], ids=lambda val: f"{val}")
def test_constant_zeros(shape: tuple):
    def module_constant_zeros():
        return jnp.zeros(shape)

    run_op_test(module_constant_zeros, [])


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    test_category=TestCategory.OP_TEST.value,
    jax_op_name="jax.numpy.ones",
    shlo_op_name="stablehlo.constant",
)
@pytest.mark.parametrize("shape", [(32, 32), (1, 1)], ids=lambda val: f"{val}")
def test_constant_ones(shape: tuple):
    def module_constant_ones():
        return jnp.ones(shape)

    run_op_test(module_constant_ones, [])


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(
    test_category=TestCategory.OP_TEST.value,
    jax_op_name="jax.numpy.array",
    shlo_op_name="stablehlo.constant",
)
@pytest.mark.xfail(reason=compile_fail("failed to legalize operation 'ttir.constant'"))
def test_constant_multi_value():
    def module_constant_multi():
        return jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)

    run_op_test(module_constant_multi, [])
