# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
from infra import Framework, run_graph_test
from utils import Category

from tests.infra.testers.compiler_config import CompilerConfig


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["conv_with_multiply.ttnn.mlir"])
@pytest.mark.xfail(
    reason="Failed to convert from TTIR to TTNN module: error: 'ttir.multiply' op result shape (1, 32, 1, 1) doesn't match expected shape after broadcasting (1, 32, 8, 8)"
)
def test_conv_with_multiply_no_bias(request):
    """Conv2d (no bias) followed by channel-wise scale — scale is absorbed into weights."""

    class ConvWithScale(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
            self.scale = nn.Parameter(torch.ones(1, 32, 1, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x) * self.scale

    run_graph_test(
        ConvWithScale().to(torch.bfloat16),
        [torch.randn(1, 16, 8, 8, dtype=torch.bfloat16)],
        framework=Framework.TORCH,
        compiler_config=CompilerConfig(optimization_level=1),
        request=request,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["conv_with_multiply.ttnn.mlir"])
@pytest.mark.xfail(
    reason="Failed to convert from TTIR to TTNN module: error: 'ttir.multiply' op result shape (1, 32, 1, 1) doesn't match expected shape after broadcasting (1, 32, 8, 8)"
)
def test_conv_with_multiply_with_bias(request):
    """Conv2d (with bias) followed by channel-wise scale — scale is absorbed into weights and bias."""

    class ConvWithScaleAndBias(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True)
            self.scale = nn.Parameter(torch.ones(1, 32, 1, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x) * self.scale

    run_graph_test(
        ConvWithScaleAndBias().to(torch.bfloat16),
        [torch.randn(1, 16, 8, 8, dtype=torch.bfloat16)],
        framework=Framework.TORCH,
        compiler_config=CompilerConfig(optimization_level=1),
        request=request,
    )
