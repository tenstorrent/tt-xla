# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
from infra import Framework, run_graph_test
from utils import Category

from tests.infra.testers.compiler_config import CompilerConfig


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["conv_add_bias.ttnn.mlir"])
def test_conv_add_bias_no_existing_bias(request):
    """Conv2d (no bias) + channel-wise add — bias is fused into the conv."""

    class ConvAddBias(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
            self.bias = nn.Parameter(torch.zeros(1, 32, 1, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x) + self.bias

    run_graph_test(
        ConvAddBias().to(torch.bfloat16),
        [torch.randn(1, 16, 8, 8, dtype=torch.bfloat16)],
        framework=Framework.TORCH,
        compiler_config=CompilerConfig(optimization_level=1),
        request=request,
    )


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["conv_add_bias.ttnn.mlir"])
def test_conv_add_bias_with_existing_bias(request):
    """Conv2d (with bias) + channel-wise add — both biases are summed and fused into the conv."""

    class ConvWithBiasAddBias(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True)
            self.extra_bias = nn.Parameter(torch.zeros(1, 32, 1, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x) + self.extra_bias

    run_graph_test(
        ConvWithBiasAddBias().to(torch.bfloat16),
        [torch.randn(1, 16, 8, 8, dtype=torch.bfloat16)],
        framework=Framework.TORCH,
        compiler_config=CompilerConfig(optimization_level=1),
        request=request,
    )
