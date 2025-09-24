# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import torch
import torch.nn as nn
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
import pytest

from infra.comparators.torch_comparator import TorchComparator
from tests.infra.comparators.comparison_config import (
    AtolConfig,
    ComparisonConfig,
    PccConfig,
)
from tests.torch.multi_chip.utils import data_parallel_inference_generic, tensor_parallel_inference_mnist

class MNISTLinear(nn.Module):
    """Simple linear MNIST model for inference testing."""
    
    def __init__(self, input_size: int = 784, hidden_size: int = 512, num_classes: int = 10, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=bias)
        # self.fc3 = nn.Linear(hidden_size, num_classes, bias=bias)
        
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(self.fc2(x))
        # x = self.fc3(x)
        return x
        # return torch.log_softmax(x, dim=1)

@pytest.mark.nightly
@pytest.mark.push
@pytest.mark.parametrize(
    "batch_size,input_size",
    [
        (32, 784),
        (128, 784),
    ],
)
def test_mnist_inference_data_parallel(batch_size: int, input_size: int):
    """Test MNIST linear inference with data parallel across multiple chips."""

    # set seed for reproducibility
    torch.manual_seed(42)

    model = MNISTLinear(input_size, 512, 10, bias=True).to(torch.bfloat16)

    # Create random input data
    input_data = torch.randn(batch_size, input_size, dtype=torch.bfloat16)

    # Run multichip inference first to get device count
    tt_output = data_parallel_inference_generic(model=model, inputs=input_data, batch_dim=0)

    # model to CPU for comparison 
    model_cpu = model.cpu()

    # CPU inference
    model_cpu.eval()
    with torch.no_grad():
        cpu_output = model_cpu(input_data)

    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(required_atol=0.05),
            pcc=PccConfig(required_pcc=0.95),
        )
    )
    comparator.compare(tt_output.cpu(), cpu_output)

@pytest.mark.nightly
@pytest.mark.push
@pytest.mark.parametrize(
    "batch_size,input_size",
    [
        (32, 784),
        # (128, 784),
    ],
)
def test_mnist_inference_tensor_parallel(batch_size: int, input_size: int):
    """Test MNIST linear inference with tensor parallel across multiple chips."""
    torch.manual_seed(42)

    model = MNISTLinear(input_size, 512, 10, bias=True).to(torch.bfloat16)

    # Random input
    input_data = torch.randn(batch_size, input_size, dtype=torch.bfloat16)

    # CPU baseline (same weights)
    model_cpu = model.cpu().eval()
    with torch.no_grad():
        cpu_output = model_cpu(input_data)

    # Run on devices (tensor parallel)
    tt_output = tensor_parallel_inference_mnist(model=model, inputs=input_data)


    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(required_atol=0.05),
            pcc=PccConfig(required_pcc=0.95),
        )
    )
    comparator.compare(tt_output.cpu(), cpu_output)