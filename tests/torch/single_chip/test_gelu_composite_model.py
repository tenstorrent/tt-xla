# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import pytest

from infra.comparators.torch_comparator import TorchComparator
from infra.comparators.comparison_config import ComparisonConfig, AtolConfig, PccConfig
from python_package.tt_torch.composite_ops import (
    enable_gelu_composite,
    disable_gelu_composite,
)


@pytest.mark.parametrize("use_composite", [False, True])
def test_gelu_composite_in_model(use_composite):
    """Test GELU composite operation in a simple model."""

    class SimpleGELUModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(64, 128, dtype=torch.bfloat16)
            self.linear2 = torch.nn.Linear(128, 64, dtype=torch.bfloat16)

        def forward(self, x):
            x = self.linear1(x)
            x = F.gelu(x)  # This will be replaced by composite if enabled
            x = self.linear2(x)
            x = F.gelu(x, approximate="tanh")  # Test tanh approximation too
            return x

    # Create model and input
    model = SimpleGELUModel()
    input_x = torch.randn(4, 64, dtype=torch.bfloat16)

    # Get golden result on CPU
    golden = model(input_x)

    # Conditionally enable composite GELU
    if use_composite:
        enable_gelu_composite()

    try:
        # Compile and run on XLA device
        device = xm.xla_device()
        model_xla = torch.compile(model.to(device), backend="tt")
        output = model_xla(input_x.to(device))

        # Compare results
        comparator = TorchComparator(
            ComparisonConfig(
                atol=AtolConfig(required_atol=0.05), pcc=PccConfig(required_pcc=0.99)
            )
        )
        comparator.compare(output, golden)

    finally:
        # Always restore original GELU if composite was enabled
        if use_composite:
            disable_gelu_composite()
