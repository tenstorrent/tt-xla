# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from tt_torch.composite_ops import composite_gelu

import torch
from torch.nn import functional as F
import torch_xla.core.xla_model as xm

from infra.comparators.torch_comparator import TorchComparator
from tests.infra.comparators.comparison_config import (
    AtolConfig,
    ComparisonConfig,
    PccConfig,
)


@pytest.mark.parametrize("approx", ["none", "tanh"])
def test_composite_gelu(approx):
    """
    Tests example model that has a composite gelu operation.
    """

    class MM(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return composite_gelu(x, approx)

    input = torch.randn(32, 32)

    model = MM()
    golden = model(input)

    device = xm.xla_device()
    model = torch.compile(model.to(device), backend="tt")

    output = model(input.to(device))
    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(required_atol=0.02), pcc=PccConfig(required_pcc=0.99)
        )
    )

    comparator.compare(output, golden)


@pytest.mark.parametrize("approx", ["none", "tanh"])
def test_composite_gelu_eager(approx):
    """
    Tests example model in eager mode that has a composite gelu operation.
    """

    class MM(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return composite_gelu(x, approx)

    input = torch.randn(32, 32)

    model = MM()
    golden = model(input)

    device = xm.xla_device()
    model = model.to(device)
    input = input.to(device)

    output = model(input).to("cpu")

    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(required_atol=0.02), pcc=PccConfig(required_pcc=0.99)
        )
    )

    comparator.compare(output, golden)


@pytest.mark.parametrize("approx", ["none", "tanh"])
def test_patched_gelu(approx):
    """
    Tests example model that has a gelu operation (replaced with our composite gelu
    operation in (tt-xla/python_package/torch_plugin_tt/__init__.py).
    """

    class MM(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return F.gelu(input=x, approximate=approx)

    input = torch.randn(32, 32)

    model = MM()
    golden = model(input)

    device = xm.xla_device()
    model = torch.compile(model.to(device), backend="tt")

    output = model(input.to(device))
    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(required_atol=0.02), pcc=PccConfig(required_pcc=0.99)
        )
    )

    comparator.compare(output, golden)


@pytest.mark.parametrize("approx", ["none", "tanh"])
def test_patched_gelu_eager(approx):
    """
    Tests example model in eager mode that has a gelu operation (replaced with our composite gelu
    operation in (tt-xla/python_package/torch_plugin_tt/__init__.py).
    """

    class MM(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return F.gelu(input=x, approximate=approx)

    input = torch.randn(32, 32)

    model = MM()
    golden = model(input)

    device = xm.xla_device()
    model = model.to(device)
    input = input.to(device)

    output = model(input).to("cpu")

    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(required_atol=0.02), pcc=PccConfig(required_pcc=0.99)
        )
    )

    comparator.compare(output, golden)
