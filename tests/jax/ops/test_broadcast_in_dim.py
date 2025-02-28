# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import jax
import jax.numpy as jnp
import pytest
from infra import run_op_test_with_random_inputs
from utils import record_unary_op_test_properties


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.parametrize("input_shapes", [[(2, 1)]], ids=lambda val: f"{val}")
def test_broadcast_in_dim(
    input_shapes: tuple, record_tt_xla_property: Callable
):
    def broadcast(a: jax.Array):
        return jnp.broadcast_to(a, (2, 4))

    record_unary_op_test_properties(
        record_tt_xla_property,
        "jax.numpy.broadcast_to",
        "stablehlo.broadcast_in_dim",
    )

    run_op_test_with_random_inputs(broadcast, input_shapes)
