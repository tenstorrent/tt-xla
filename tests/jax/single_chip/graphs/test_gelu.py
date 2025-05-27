# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import pytest
from infra import run_graph_test_with_random_inputs
from tests.utils import incorrect_result
from tests.utils import Category

import flax.linen as nn


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.parametrize(
    "x_shape",
    [
        (32, 32),
        (64, 64),
    ],
)
@pytest.mark.parametrize(
    "approximate",
    [
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                reason=incorrect_result(
                    "Atol comparison failed. Calculated: atol=nan. Required: atol=0.16."
                )
            ),
        ),
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                reason=incorrect_result(
                    "Allclose comparison failed. Required: atol=0.01, rtol=0.01."
                )
            ),
        ),
    ],
)
def test_gelu(x_shape, approximate):
    def gelu(x: jax.Array) -> jax.Array:
        return nn.gelu(x, approximate=approximate)

    run_graph_test_with_random_inputs(gelu, [x_shape])
