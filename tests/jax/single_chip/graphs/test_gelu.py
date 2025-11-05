# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import flax.linen as nn
import jax
import pytest
from infra import run_graph_test_with_random_inputs
from utils import Category, incorrect_result


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
        False,
        True,
    ],
)
def test_gelu(x_shape, approximate):
    def gelu(x: jax.Array) -> jax.Array:
        return nn.gelu(x, approximate=approximate)

    run_graph_test_with_random_inputs(gelu, [x_shape])
