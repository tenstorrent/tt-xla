# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_graph_test_with_random_inputs
from utils import Category


@pytest.mark.extended
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["reshape_broadcast_reshape_to_repeat_interleave.ttnn.mlir"])
def test_reshape_broadcast_reshape_to_repeat_interleave(request):
    def repeat_interleave_pattern(x: torch.Tensor) -> torch.Tensor:
        return (
            x.reshape(11, 3, 1, 32, 64).expand(11, 3, 5, 32, 64).reshape(11, 15, 32, 64)
        )

    run_graph_test_with_random_inputs(
        repeat_interleave_pattern,
        [(11, 3, 32, 64)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.extended
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["reshape_broadcast_reshape_to_repeat.ttnn.mlir"])
def test_reshape_broadcast_reshape_to_repeat(request):
    def repeat_pattern(x: torch.Tensor) -> torch.Tensor:
        return (
            x.reshape(11, 1, 3, 32, 64).expand(11, 5, 3, 32, 64).reshape(11, 15, 32, 64)
        )

    run_graph_test_with_random_inputs(
        repeat_pattern,
        [(11, 3, 32, 64)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )
