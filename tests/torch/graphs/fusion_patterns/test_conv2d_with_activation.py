# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from infra import Framework, run_graph_test
from utils import Category


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["conv2d_with_activation_relu.ttnn.mlir"])
def test_conv2d_with_relu(request):
    class ConvRelu(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.relu(self.conv(x))

    run_graph_test(
        ConvRelu().to(torch.bfloat16),
        [torch.randn(1, 16, 8, 8, dtype=torch.bfloat16)],
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["conv2d_with_activation_relu6.ttnn.mlir"])
def test_conv2d_with_relu6(request):
    class ConvRelu6(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.relu6(self.conv(x))

    run_graph_test(
        ConvRelu6().to(torch.bfloat16),
        [torch.randn(1, 16, 8, 8, dtype=torch.bfloat16)],
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["conv2d_with_activation_silu.ttnn.mlir"])
def test_conv2d_with_silu(request):
    class ConvSilu(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.silu(self.conv(x))

    run_graph_test(
        ConvSilu().to(torch.bfloat16),
        [torch.randn(1, 16, 8, 8, dtype=torch.bfloat16)],
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.filecheck(["conv2d_with_activation_sigmoid.ttnn.mlir"])
def test_conv2d_with_sigmoid(request):
    class ConvSigmoid(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.conv(x))

    run_graph_test(
        ConvSigmoid().to(torch.bfloat16),
        [torch.randn(1, 16, 8, 8, dtype=torch.bfloat16)],
        framework=Framework.TORCH,
        request=request,
    )
