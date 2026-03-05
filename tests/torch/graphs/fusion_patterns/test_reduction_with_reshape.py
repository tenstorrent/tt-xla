# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_graph_test_with_random_inputs
from utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["reduction_with_reshape.ttnn.mlir"])
@pytest.mark.parametrize("reduce_fn", [torch.mean, torch.sum], ids=["mean", "sum"])
def test_reduction_with_reshape(reduce_fn, request):
    def reduction_with_reshape(x: torch.Tensor) -> torch.Tensor:
        return reduce_fn(x, dim=2).unsqueeze(2)

    run_graph_test_with_random_inputs(
        reduction_with_reshape,
        [(1, 8, 16, 32)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )
