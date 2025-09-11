# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch_xla.core.xla_model as xm
import pytest

from infra.comparators.torch_comparator import TorchComparator
from tests.infra.comparators.comparison_config import (
    AtolConfig,
    ComparisonConfig,
    PccConfig,
)


@pytest.mark.nightly
@pytest.mark.push
@pytest.mark.parametrize(
    "input_shape",
    [
        (2048, 2048),
        (1024, 512),
        (512, 1024),
    ],
)
def test_all_reduce(input_shape: tuple):
    """Test torch XLA all_reduce operation for data parallel patterns."""
    
    class AllReduceModule(torch.nn.Module):
        def forward(self, x):
            # Perform all_reduce SUM operation (equivalent to JAX psum)
            # This is what DDP uses to aggregate gradients across devices
            return xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Create input tensor
    input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    
    # Get golden output (on single device, all_reduce should be identity)
    model = AllReduceModule()
    golden = model(input_tensor)
    
    # Run on TT device
    device = xm.xla_device()
    model = torch.compile(model.to(device), backend="tt")
    output = model(input_tensor.to(device))
    
    # Compare results
    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(required_atol=0.02),
            pcc=PccConfig(required_pcc=0.99),
        )
    )
    comparator.compare(output, golden)


@pytest.mark.nightly 
@pytest.mark.push
@pytest.mark.parametrize(
    "input_shape",
    [
        (8192, 784),
        (4096, 512),
    ],
)
def test_all_gather(input_shape: tuple):
    """Test torch XLA all_gather operation for data parallel patterns."""
    
    class AllGatherModule(torch.nn.Module):
        def forward(self, x):
            # Perform all_gather operation (broadcast-like for single device)
            return xm.all_gather(x)
    
    # Create input tensor
    input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    
    # Get golden output (on single device, all_gather should be identity)
    model = AllGatherModule()
    golden = model(input_tensor)
    
    # Run on TT device
    device = xm.xla_device()
    model = torch.compile(model.to(device), backend="tt")
    output = model(input_tensor.to(device))
    
    # Compare results
    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(required_atol=0.02),
            pcc=PccConfig(required_pcc=0.99),
        )
    )
    comparator.compare(output, golden)