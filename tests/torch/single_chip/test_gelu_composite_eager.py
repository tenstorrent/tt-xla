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


def test_gelu_composite_eager_mode():
    """Test GELU composite operation in eager mode (no torch.compile)."""

    class SimpleGELUModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(64, 128, dtype=torch.bfloat16)
            self.linear2 = torch.nn.Linear(128, 64, dtype=torch.bfloat16)

        def forward(self, x):
            x = self.linear1(x)
            x = F.gelu(x)  # This will be replaced by composite
            x = self.linear2(x)
            x = F.gelu(x, approximate="tanh")  # Test tanh approximation too
            return x

    # Create model and input
    model = SimpleGELUModel()
    input_x = torch.randn(4, 64, dtype=torch.bfloat16)

    # Get golden result on CPU
    golden = model(input_x)

    # Run on XLA device WITHOUT torch.compile
    device = xm.xla_device()
    model_xla = model.to(device)
    input_x_xla = input_x.to(device)

    # Test without composite
    output_normal = model_xla(input_x_xla)
    xm.mark_step()  # Ensure XLA execution completes

    # Convert to CPU for comparison
    output_normal_cpu = output_normal.to("cpu")

    # Compare normal GELU
    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(required_atol=0.05), pcc=PccConfig(required_pcc=0.99)
        )
    )
    comparator.compare(output_normal_cpu, golden)
    print("Normal GELU in eager mode passed")

    # Now test with composite enabled
    enable_gelu_composite()
    try:
        # Use the SAME model (model_xla) that's already on device
        # This ensures we're comparing apples to apples

        # Run with composite GELU
        output_composite = model_xla(input_x_xla)
        xm.mark_step()  # Ensure XLA execution completes

        output_composite_cpu = output_composite.to("cpu")

        # Compare composite GELU with golden
        comparator.compare(output_composite_cpu, golden)
        print("Composite GELU in eager mode passed")

        # Also compare that normal and composite produce similar results
        comparator.compare(output_composite_cpu, output_normal_cpu)
        print("Normal and composite GELU outputs match")

    finally:
        disable_gelu_composite()
