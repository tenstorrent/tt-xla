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
@pytest.mark.filecheck(["repvgg_conv_sum.ttnn.mlir"])
def test_repvgg_conv_sum_fusion(request):
    def repvgg_block(
        x: torch.Tensor, w3x3: torch.Tensor, w1x1: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.conv2d(
            x, w3x3, padding=1
        ) + torch.nn.functional.conv2d(x, w1x1)

    run_graph_test_with_random_inputs(
        repvgg_block,
        [(1, 16, 32, 32), (16, 16, 3, 3), (16, 16, 1, 1)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        request=request,
    )
