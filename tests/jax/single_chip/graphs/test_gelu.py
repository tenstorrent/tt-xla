# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax.nn as nn
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
        *([(32, 32)] * 100),
        [(64, 64)],
    ],
)
@pytest.mark.parametrize(
    "approximate",
    [
        pytest.param(
            False,
            # marks=pytest.mark.xfail(
            #     reason=incorrect_result(
            #         "PCC comparison failed. Calculated: pcc=nan. Required: pcc=0.99. "
            #         "https://github.com/tenstorrent/tt-xla/issues/379"
            #     )
            # ),
        ),
        pytest.param(True),
    ],
)
def test_gelu(x_shape, approximate):
    def gelu(x: jax.Array) -> jax.Array:
        return nn.gelu(x, approximate=approximate)

    run_graph_test_with_random_inputs(gelu, [x_shape])
