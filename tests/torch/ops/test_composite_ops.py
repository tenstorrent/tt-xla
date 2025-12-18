# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.comparators.torch_comparator import TorchComparator
from infra.utilities.types import Framework
from torch.nn import functional as F
from tt_torch.composite_ops import (
    composite_gelu,
    composite_layer_norm,
    composite_rms_norm,
)

from tests.infra.comparators.comparison_config import ComparisonConfig
from tests.infra.testers.single_chip.graph.graph_tester import run_graph_test
from tests.infra.testers.single_chip.op.op_tester import run_op_test_with_random_inputs
from tests.utils import parametrize_arch


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


@parametrize_arch(["single_device", "dual_chip"])
@pytest.mark.parametrize("use_weight", [True, False])
def test_rmsnorm(use_weight, arch):

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

    input_tensor = input_tensor.to(device)
    weight = weight.to(device) if use_weight else None

    if arch == "dual_chip":
        # Set SPMD mode and get number of devices.
        os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
        xr.use_spmd()

        num_devices = xr.global_runtime_device_count()

        # Create a mesh.
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = xs.Mesh(device_ids, mesh_shape, ("model", "batch"))

        # Mark sharding for inputs along batch dimension.
        xs.mark_sharding(input_tensor, mesh, ("batch", None))
        if use_weight:
            xs.mark_sharding(weight, mesh, (None,))

    output = model(input_tensor, weight)

    comparator = TorchComparator(ComparisonConfig())
    comparator.compare(output, golden)


@pytest.mark.single_device
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


@pytest.mark.single_device
@pytest.mark.parametrize("elementwise_affine", [True, False])
@pytest.mark.parametrize(
    "batch_size, sentence_length, embedding_dim",
    [(1, 32, 32), (1, 197, 768), (1, 1024, 768)],
)
def test_patched_layer_norm_module(
    elementwise_affine, batch_size, sentence_length, embedding_dim
):
    class LayerNormModel(torch.nn.Module):
        def __init__(self, embedding_dim):
            super().__init__()
            self.ln = nn.LayerNorm(embedding_dim, elementwise_affine=elementwise_affine)

        def forward(self, x):
            return self.ln(x)

    options = {"tt_enable_composite_ops": True}

    input_tensor = torch.randn(
        batch_size, sentence_length, embedding_dim, dtype=torch.bfloat16
    )

    model = LayerNormModel(embedding_dim)

    run_graph_test(
        model,
        [input_tensor],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        torch_options=options,
    )


@pytest.mark.single_device
@pytest.mark.parametrize("use_weight", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize(
    "batch_size, sentence_length, embedding_dim",
    [(1, 32, 32), (1, 197, 768), (1, 1024, 768)],
)
def test_patched_layer_norm_functional(
    use_weight, use_bias, batch_size, sentence_length, embedding_dim
):

    class LayerNormModel(torch.nn.Module):
        def __init__(self, normalized_shape):
            super().__init__()
            self.normalized_shape = normalized_shape

        def forward(self, x, weight=None, bias=None):
            return F.layer_norm(x, (self.normalized_shape,), weight, bias, eps=1e-5)

    options = {"tt_enable_composite_ops": True}

    input_tensor = torch.randn(
        batch_size, sentence_length, embedding_dim, dtype=torch.bfloat16
    )
    weight = torch.randn(embedding_dim, dtype=torch.bfloat16) if use_weight else None
    bias = torch.randn(embedding_dim, dtype=torch.bfloat16) if use_bias else None

    model = LayerNormModel(embedding_dim)

    run_graph_test(
        model,
        [input_tensor, weight, bias],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        torch_options=options,
    )


@pytest.mark.single_device
@pytest.mark.parametrize("use_weight", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize(
    "batch_size, sentence_length, embedding_dim",
    [(1, 32, 32), (1, 197, 768), (1, 1024, 768)],
)
def test_composite_layer_norm(
    use_weight, use_bias, batch_size, sentence_length, embedding_dim
):

    class LayerNormModel(torch.nn.Module):
        def __init__(self, normalized_shape):
            super().__init__()
            self.normalized_shape = normalized_shape

        def forward(self, x, weight=None, bias=None):
            return composite_layer_norm(
                x, self.normalized_shape, weight, bias, eps=1e-5
            )

    options = {"tt_enable_composite_ops": False}

    input_tensor = torch.randn(
        batch_size, sentence_length, embedding_dim, dtype=torch.bfloat16
    )
    weight = torch.randn(embedding_dim, dtype=torch.bfloat16) if use_weight else None
    bias = torch.randn(embedding_dim, dtype=torch.bfloat16) if use_bias else None

    model = LayerNormModel(embedding_dim)

    # Disable inplace buffers for inductor compilation
    # so that we can compare the results with the golden model.
    with torch._inductor.config.patch({"inplace_buffers": False}):
        run_graph_test(
            model,
            [input_tensor, weight, bias],
            comparison_config=ComparisonConfig(),
            framework=Framework.TORCH,
            torch_options=options,
        )
