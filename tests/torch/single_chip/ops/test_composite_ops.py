# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_xla.core.xla_model as xm
from infra.comparators.torch_comparator import TorchComparator
from infra.utilities.types import Framework
from torch.nn import RMSNorm
from torch.nn import functional as F
from tt_torch.composite_ops import composite_gelu, composite_rms_norm

from tests.infra.comparators.comparison_config import ComparisonConfig
from tests.infra.testers.single_chip.op.op_tester import run_op_test_with_random_inputs


@pytest.mark.parametrize("approx", ["none", "tanh"])
def test_composite_gelu(approx):
    """
    Tests example model that has a composite gelu operation.
    """

    class MM(torch.nn.Module):
        def forward(self, x):
            return composite_gelu(x, approx)

    input = torch.randn(32, 32)

    model = MM()
    golden = model(input)

    device = xm.xla_device()
    model = torch.compile(model.to(device), backend="tt")

    output = model(input.to(device))

    comparator = TorchComparator(ComparisonConfig())
    comparator.compare(output, golden)


@pytest.mark.parametrize("approx", ["none", "tanh"])
def test_composite_gelu_eager(approx):
    """
    Tests example model in eager mode that has a composite gelu operation.
    """

    class MM(torch.nn.Module):
        def forward(self, x):
            return composite_gelu(x, approx)

    input = torch.randn(32, 32)

    model = MM()
    golden = model(input)

    device = xm.xla_device()
    model = model.to(device)
    input = input.to(device)

    output = model(input).to("cpu")

    comparator = TorchComparator(ComparisonConfig())
    comparator.compare(output, golden)


@pytest.mark.parametrize("approx", ["none", "tanh"])
def test_patched_gelu(approx):
    """
    Tests example model that has a gelu operation (replaced with our composite gelu
    operation in (tt-xla/python_package/torch_plugin_tt/__init__.py).
    """

    class MM(torch.nn.Module):
        def forward(self, x):
            return F.gelu(input=x, approximate=approx)

    input = torch.randn(32, 32)

    model = MM()
    golden = model(input)

    device = xm.xla_device()
    model = torch.compile(model.to(device), backend="tt")

    output = model(input.to(device))

    comparator = TorchComparator(ComparisonConfig())
    comparator.compare(output, golden)


@pytest.mark.parametrize("approx", ["none", "tanh"])
def test_patched_gelu_eager(approx):
    """
    Tests example model in eager mode that has a gelu operation (replaced with our composite gelu
    operation in (tt-xla/python_package/torch_plugin_tt/__init__.py).
    """

    class MM(torch.nn.Module):
        def forward(self, x):
            return F.gelu(input=x, approximate=approx)

    input = torch.randn(32, 32)

    model = MM()
    golden = model(input)

    device = xm.xla_device()
    model = model.to(device)
    input = input.to(device)

    output = model(input).to("cpu")

    comparator = TorchComparator(ComparisonConfig())
    comparator.compare(output, golden)


@pytest.mark.parametrize("approx", ["none", "tanh"])
def test_patched_gelu_op_test(approx):
    """
    Tests torch gelu operation (replaced with our composite gelu operation in
    (tt-xla/python_package/torch_plugin_tt/__init__.py) using run_op_test_with_random_inputs utility.
    """

    def gelu_with_approx(x):
        return F.gelu(x, approximate=approx)

    run_op_test_with_random_inputs(
        gelu_with_approx, [(32, 32)], framework=Framework.TORCH
    )


def test_composite_rms_norm():

    input = torch.randn(32, 32)

    class MM(torch.nn.Module):
        def forward(self, x):
            return torch.rms_norm(x, input.shape)

    # When created like a module, we can not create composite op from torch.rms_norm,
    # because there is not torch.rms_norm node in a fx graph, so we need to create op manually within MM module.
    # model = RMSNorm(normalized_shape=input.shape)
    model = MM()
    golden = model(input)

    device = xm.xla_device()
    model = torch.compile(model.to(device), backend="tt")

    output = model(input.to(device))

    comparator = TorchComparator(ComparisonConfig())
    comparator.compare(output, golden)
