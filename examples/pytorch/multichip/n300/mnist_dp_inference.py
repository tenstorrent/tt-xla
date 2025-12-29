# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.evaluators import (
    AtolConfig,
    ComparisonConfig,
    PccConfig,
    TorchComparisonEvaluator,
)
from infra.utilities.torch_multichip_utils import enable_spmd
from torch_xla.distributed.spmd import Mesh
from utils import data_parallel_inference_generic


# these examples should be moved to the test infra once this issue is closed: https://github.com/tenstorrent/tt-xla/issues/1822
class MNISTLinear(nn.Module):
    """Simple linear MNIST model for inference testing."""

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 512,
        num_classes: int = 10,
        bias: bool = True,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.fc3 = nn.Linear(hidden_size, num_classes, bias=bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.log_softmax(x, dim=1)


@pytest.mark.parametrize(
    "batch_size,input_size",
    [
        (32, 784),
        (128, 784),
    ],
)
def test_mnist_inference_data_parallel(batch_size: int, input_size: int):
    """Test MNIST linear inference with data parallel across multiple chips."""
    xr.set_device_type("TT")
    enable_spmd()

    # set seed for reproducibility
    torch.manual_seed(42)

    model = MNISTLinear(input_size, 512, 10, bias=True).to(torch.bfloat16)

    # Create random input data
    input_data = torch.randn(batch_size, input_size, dtype=torch.bfloat16)

    # Run multichip inference first to get device count
    tt_output = data_parallel_inference_generic(
        model=model, inputs=input_data, batch_dim=0
    )

    # model to CPU for comparison
    model_cpu = model.cpu()

    # CPU inference
    model_cpu.eval()
    with torch.no_grad():
        cpu_output = model_cpu(input_data)

    comparator = TorchComparisonEvaluator(
        ComparisonConfig(
            atol=AtolConfig(required_atol=0.05),
            pcc=PccConfig(required_pcc=0.99),
        )
    )
    comparator.compare(tt_output.cpu(), cpu_output)
