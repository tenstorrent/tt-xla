# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import pytest
from infra import run_graph_test_with_random_inputs
from utils import Category, failed_runtime


def conditionally_skip(x_shape: tuple, axis: int):
    """
    Helper function which checks x_shape-axis combination and skips if unsupported for some
    reason.

    Extracted here in order not to pollute the test function.
    """
    if axis != len(x_shape) - 1:
        pytest.xfail(
            failed_runtime(
                "Softmax axis which is not the last dimension is not supported. "
                "TT-metal issue: https://github.com/tenstorrent/tt-metal/issues/21906"
            )
        )


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.parametrize(
    ["x_shape", "axis"],
    [
        [(32, 32), 0],
        [(32, 32), 1],
        [(64, 64), 0],
        [(64, 64), 1],
    ],
)
def test_softmax(x_shape: tuple, axis: int):
    def softmax(x: jax.Array) -> jax.Array:
        return jax.nn.softmax(x, axis=axis)

    conditionally_skip(x_shape, axis)

    run_graph_test_with_random_inputs(softmax, [x_shape])
