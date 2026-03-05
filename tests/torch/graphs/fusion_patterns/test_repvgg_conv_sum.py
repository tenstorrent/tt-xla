# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, run_graph_test
from utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["repvgg_conv_sum.ttnn.mlir"])
def test_repvgg_conv_sum_fusion(request):
    class RepVGGBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv3x3 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
            self.conv1x1 = torch.nn.Conv2d(16, 16, kernel_size=1, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv3x3(x) + self.conv1x1(x)

    model = RepVGGBlock()
    x = torch.randn(1, 16, 32, 32, dtype=torch.bfloat16)

    run_graph_test(model, [x], framework=Framework.TORCH, request=request)
