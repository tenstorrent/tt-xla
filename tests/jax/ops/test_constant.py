# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import jax.numpy as jnp
import pytest
from infra import run_op_test
from utils import record_op_test_properties


@pytest.mark.parametrize("shape", [(32, 32), (1, 1)], ids=lambda val: f"{val}")
def test_constant_zeros(shape: tuple, record_tt_xla_property: Callable):
    def module_constant_zeros():
        return jnp.zeros(shape)

    record_op_test_properties(
        record_tt_xla_property,
        "Constant op",
        "jax.numpy.zeros",
        "stablehlo.constant",
    )

    run_op_test(module_constant_zeros, [])


@pytest.mark.parametrize("shape", [(32, 32), (1, 1)], ids=lambda val: f"{val}")
def test_constant_ones(shape: tuple, record_tt_xla_property: Callable):
    def module_constant_ones():
        return jnp.ones(shape)

    record_op_test_properties(
        record_tt_xla_property,
        "Constant op",
        "jax.numpy.ones",
        "stablehlo.constant",
    )

    run_op_test(module_constant_ones, [])


@pytest.mark.xfail(reason="failed to legalize operation 'ttir.constant'")
def test_constant_multi_value(record_tt_xla_property: Callable):
    def module_constant_multi():
        return jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)

    record_op_test_properties(
        record_tt_xla_property,
        "Constant op",
        "jax.numpy.array",
        "stablehlo.constant",
    )

    run_op_test(module_constant_multi, [])
