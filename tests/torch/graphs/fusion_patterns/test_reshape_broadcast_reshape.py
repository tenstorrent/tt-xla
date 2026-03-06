# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_graph_test_with_random_inputs
from utils import Category


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["reshape_broadcast_reshape_to_repeat_interleave.ttnn.mlir"])
def test_reshape_broadcast_reshape_to_repeat_interleave(request):
    # insert 1 after kv_heads: (1,2,32,64) -> (1,2,1,32,64) -> expand -> (1,8,32,64)
    # changedDim(1) == insertedDim(2) - 1  =>  repeat_interleave
    def repeat_interleave_pattern(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(1, 2, 1, 32, 64).expand(1, 2, 4, 32, 64).reshape(1, 8, 32, 64)

    run_graph_test_with_random_inputs(
        repeat_interleave_pattern,
        [(1, 2, 32, 64)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["reshape_broadcast_reshape_to_repeat.ttnn.mlir"])
def test_reshape_broadcast_reshape_to_repeat(request):
    # insert 1 before kv_heads: (1,2,32,64) -> (1,1,2,32,64) -> expand -> (1,8,32,64)
    # changedDim(1) == insertedDim(1)  =>  repeat
    def repeat_pattern(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(1, 1, 2, 32, 64).expand(1, 4, 2, 32, 64).reshape(1, 8, 32, 64)

    run_graph_test_with_random_inputs(
        repeat_pattern,
        [(1, 2, 32, 64)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )
