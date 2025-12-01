# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_xla.core.xla_model as xm
from infra.comparators.torch_comparator import TorchComparator
from infra.utilities.types import Framework
from torch.nn import functional as F
from tt_torch.composite_ops import composite_gelu, composite_rms_norm

from tests.infra.comparators.comparison_config import ComparisonConfig
from tests.infra.testers.single_chip.op.op_tester import run_op_test_with_random_inputs


@pytest.mark.single_device
@pytest.mark.parametrize("approx", ["none", "tanh"])
def test_composite_gelu(approx):
    """
    Tests example model that has a composite gelu operation.
    """

    class MM(torch.nn.Module):
        def forward(self, x):
            return composite_gelu(x, approx)

    options = {"tt_enable_composite_ops": False}

    input = torch.randn(32, 32)

    model = MM()
    golden = model(input)

    device = xm.xla_device()
    model = torch.compile(model.to(device), backend="tt", options=options)

    output = model(input.to(device))

    comparator = TorchComparator(ComparisonConfig())
    comparator.compare(output, golden)


@pytest.mark.single_device
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


@pytest.mark.single_device
@pytest.mark.parametrize("approx", ["none", "tanh"])
def test_patched_gelu(approx):
    """
    Tests example model that has a gelu operation (replaced with our composite gelu
    operation in (tt-xla/python_package/torch_plugin_tt/__init__.py).
    """

    class MM(torch.nn.Module):
        def forward(self, x):
            return F.gelu(input=x, approximate=approx)

    options = {"tt_enable_composite_ops": True}

    input = torch.randn(32, 32)

    model = MM()
    golden = model(input)

    device = xm.xla_device()
    model = torch.compile(model.to(device), backend="tt", options=options)

    output = model(input.to(device))

    comparator = TorchComparator(ComparisonConfig())
    comparator.compare(output, golden)


@pytest.mark.single_device
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


@pytest.mark.single_device
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


@pytest.mark.parametrize("use_weight", [True, False])
def test_rmsnorm(use_weight):

    class RMSNormModel(torch.nn.Module):
        def __init__(self, normalized_shape):
            super().__init__()
            self.normalized_shape = normalized_shape

        def forward(self, x, weight):
            return torch.nn.functional.rms_norm(x, self.normalized_shape, weight)

    options = {"tt_enable_composite_ops": True}

    normalized_shape = (32,)
    input_shape = (4, 32)
    input_tensor = torch.randn(input_shape)

    weight = torch.randn(normalized_shape) if use_weight else None

    model = RMSNormModel(normalized_shape)
    golden = model(input_tensor, weight if use_weight else None)

    device = xm.xla_device()
    model = torch.compile(model.to(device), backend="tt", options=options)

    output = model(input_tensor.to(device), weight.to(device) if use_weight else None)

    comparator = TorchComparator(ComparisonConfig())
    comparator.compare(output, golden)


@pytest.mark.parametrize("use_weight", [True, False])
def test_composite_rms_norm(use_weight):
    """
    Tests example model that has a composite RMS norm operation.
    """

    class RMSNormModel(torch.nn.Module):
        def __init__(self, normalized_shape):
            super().__init__()
            self.normalized_shape = normalized_shape

        def forward(self, x, weight):
            return composite_rms_norm(x, self.normalized_shape, weight)

    options = {"tt_enable_composite_ops": False}

    normalized_shape = (32,)
    input_shape = (4, 32)
    input_tensor = torch.randn(input_shape)
    weight = torch.randn(normalized_shape) if use_weight else None

    model = RMSNormModel(normalized_shape)
    golden = model(input_tensor, weight if use_weight else None)

    device = xm.xla_device()
    model = torch.compile(model.to(device), backend="tt", options=options)
    output = model(input_tensor.to(device), weight.to(device) if use_weight else None)

    comparator = TorchComparator(ComparisonConfig())
    comparator.compare(output, golden)
