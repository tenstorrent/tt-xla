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
@pytest.mark.filecheck(["matmul_with_bias.ttnn.mlir"])
@pytest.mark.xfail(reason="Turns out this fusion is broken?")
def test_matmul_with_bias_fusion(request):
    def matmul_with_bias(
        x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.linear(x, weight, bias)

    run_graph_test_with_random_inputs(
        matmul_with_bias,
        [(1, 32, 64), (128, 64), (128,)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )
