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
@pytest.mark.parametrize(
    ["in_shape", "out_shape"],
    [
        ((8, 32, 256), (2, 4, 32, 256)),
        ((8, 32, 32), (1, 2, 4, 32, 32)),
        ((8192, 128), (1, 256, 32, 128)),
    ],
    ids=lambda val: f"{val}",
)
def test_reshape(in_shape: tuple, out_shape: tuple, record_tt_xla_property: Callable):
    def reshape(x: jax.Array):
        return jnp.reshape(x, out_shape)

    record_unary_op_test_properties(
        record_tt_xla_property,
        "jax.numpy.reshape",
        "stablehlo.reshape",
    )

    run_op_test_with_random_inputs(reshape, [in_shape])
