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
from infra.evaluators import TorchComparisonEvaluator
from infra.utilities.types import Framework
from torch.nn import functional as F
from tt_torch.composite_ops import (
    composite_gelu,
    composite_layer_norm,
    composite_rms_norm,
    composite_topk,
    composite_topk_indices,
    composite_topk_values,
)

from tests.infra.evaluators.evaluation_config import ComparisonConfig
from tests.infra.testers.single_chip.graph.graph_tester import run_graph_test


@pytest.mark.single_device
@pytest.mark.parametrize("approx", ["none", "tanh"])
def test_composite_gelu(approx):
    class GeluModel(torch.nn.Module):
        def forward(self, x):
            return composite_gelu(x, approx)

    options = {"tt_enable_composite_ops": False}

    input = torch.randn(32, 32)
    model = GeluModel()

    # Disable inplace buffers for inductor compilation
    # so that we can compare the results with the golden model.
    with torch._inductor.config.patch({"inplace_buffers": False}):
        run_graph_test(
            model,
            [input],
            comparison_config=ComparisonConfig(),
            framework=Framework.TORCH,
            torch_options=options,
        )


@pytest.mark.single_device
@pytest.mark.parametrize("approx", ["none", "tanh"])
def test_patched_gelu_functional(approx):
    class GeluModel(torch.nn.Module):
        def forward(self, x):
            return F.gelu(input=x, approximate=approx)

    options = {"tt_enable_composite_ops": True}

    input = torch.randn(32, 32)
    model = GeluModel()

    run_graph_test(
        model,
        [input],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        torch_options=options,
    )


@pytest.mark.single_device
@pytest.mark.parametrize("use_weight", [True, False])
@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size", [(1, 32, 32), (1, 128, 768), (1, 1024, 768)]
)
def test_patched_rms_norm_functional_single_device(
    use_weight, batch_size, seq_len, hidden_size
):

    class RMSNormModel(torch.nn.Module):
        def __init__(self, normalized_shape):
            super().__init__()
            self.normalized_shape = normalized_shape

        def forward(self, x, weight):
            return torch.nn.functional.rms_norm(x, (self.normalized_shape,), weight)

    options = {"tt_enable_composite_ops": True}

    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    weight = torch.randn(hidden_size) if use_weight else None
    model = RMSNormModel(hidden_size)

    run_graph_test(
        model,
        [input_tensor, weight],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        torch_options=options,
    )


@pytest.mark.dual_chip
@pytest.mark.parametrize("use_weight", [True, False])
@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size", [(1, 32, 32), (1, 128, 768), (1, 1024, 768)]
)
def test_patched_rms_norm_functional_batch_parallel(
    use_weight, batch_size, seq_len, hidden_size
):

    class RMSNormModel(torch.nn.Module):
        def __init__(self, normalized_shape):
            super().__init__()
            self.normalized_shape = normalized_shape

        def forward(self, x, weight):
            return torch.nn.functional.rms_norm(x, (self.normalized_shape,), weight)

    options = {"tt_enable_composite_ops": True}

    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    weight = torch.randn(hidden_size) if use_weight else None
    model = RMSNormModel(hidden_size)

    # Create a mesh.
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = xs.Mesh(device_ids, mesh_shape, ("model", "batch"))

    # Mark sharding for inputs along batch dimension.
    shard_specs = {}
    shard_specs[input_tensor] = ("batch", None)
    if use_weight:
        shard_specs[weight] = (None,)

    run_graph_test(
        model,
        [input_tensor, weight],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        torch_options=options,
        mesh=mesh,
        shard_spec_fn=shard_specs,
    )


@pytest.mark.single_device
@pytest.mark.parametrize("use_weight", [True, False])
@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size", [(1, 32, 32), (1, 128, 768), (1, 1024, 768)]
)
def test_composite_rms_norm(use_weight, batch_size, seq_len, hidden_size):

    class RMSNormModel(torch.nn.Module):
        def __init__(self, normalized_shape):
            super().__init__()
            self.normalized_shape = normalized_shape

        def forward(self, x, weight):
            return composite_rms_norm(x, (self.normalized_shape,), weight)

    options = {"tt_enable_composite_ops": False}

    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    weight = torch.randn(hidden_size) if use_weight else None
    model = RMSNormModel(hidden_size)

    # Disable inplace buffers for inductor compilation
    # so that we can compare the results with the golden model.
    with torch._inductor.config.patch({"inplace_buffers": False}):
        run_graph_test(
            model,
            [input_tensor, weight],
            comparison_config=ComparisonConfig(),
            framework=Framework.TORCH,
            torch_options=options,
        )


@pytest.mark.single_device
@pytest.mark.parametrize("elementwise_affine", [True, False])
@pytest.mark.parametrize(
    "batch_size, seq_len, embedding_dim",
    [(1, 32, 32), (1, 197, 768), (1, 1024, 768)],
)
def test_patched_layer_norm_module(
    elementwise_affine, batch_size, seq_len, embedding_dim
):
    class LayerNormModel(torch.nn.Module):
        def __init__(self, embedding_dim):
            super().__init__()
            self.ln = nn.LayerNorm(embedding_dim, elementwise_affine=elementwise_affine)

        def forward(self, x):
            return self.ln(x)

    options = {"tt_enable_composite_ops": True}

    input_tensor = torch.randn(batch_size, seq_len, embedding_dim, dtype=torch.bfloat16)

    model = LayerNormModel(embedding_dim)

    run_graph_test(
        model,
        [input_tensor],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        torch_options=options,
    )


@pytest.mark.single_device
@pytest.mark.parametrize(
    "use_weight, use_bias", [(True, True), (True, False), (False, False)]
)
@pytest.mark.parametrize(
    "batch_size, seq_len, embedding_dim",
    [(1, 32, 32), (1, 197, 768), (1, 1024, 768)],
)
def test_patched_layer_norm_functional(
    use_weight, use_bias, batch_size, seq_len, embedding_dim
):

    class LayerNormModel(torch.nn.Module):
        def __init__(self, normalized_shape):
            super().__init__()
            self.normalized_shape = normalized_shape

        def forward(self, x, weight=None, bias=None):
            return F.layer_norm(x, (self.normalized_shape,), weight, bias, eps=1e-5)

    options = {"tt_enable_composite_ops": True}

    input_tensor = torch.randn(batch_size, seq_len, embedding_dim, dtype=torch.bfloat16)
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
@pytest.mark.parametrize("k", [1, 5, 16])
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("largest", [True, False])
def test_composite_topk(k, dim, largest):
    if dim == 0 and k == 16:
        pytest.skip()
    class TopkModel(torch.nn.Module):
        def forward(self, x):
            return composite_topk(x, k, dim, largest)

    options = {"tt_enable_composite_ops": False}

    input = torch.randn(8, 16, dtype=torch.bfloat16)
    model = TopkModel()

    with torch._inductor.config.patch({"inplace_buffers": False}):
        run_graph_test(
            model,
            [input],
            comparison_config=ComparisonConfig(),
            framework=Framework.TORCH,
            torch_options=options,
        )


@pytest.mark.single_device
@pytest.mark.parametrize("k", [1, 5, 16])
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("largest", [True, False])
def test_patched_topk(k, dim, largest):
    if dim == 0 and k == 16:
        pytest.skip()
    class TopkModel(torch.nn.Module):
        def forward(self, x):
            return torch.topk(x, k, dim, largest)

    options = {"tt_enable_composite_ops": True}

    input = torch.randn(8, 16)
    model = TopkModel()

    run_graph_test(
        model,
        [input],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        torch_options=options,
    )


@pytest.mark.single_device
@pytest.mark.parametrize("k", [1, 5, 16])
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("largest", [True, False])
def test_composite_topk_values(k, dim, largest):
    """Test single-output composite that returns only the values tensor."""
    if dim == 0 and k == 16:
        pytest.skip()

    class TopkValuesModel(torch.nn.Module):
        def forward(self, x):
            return composite_topk_values(x, k, dim, largest)

    options = {"tt_enable_composite_ops": False}

    input = torch.randn(8, 16)
    model = TopkValuesModel()

    with torch._inductor.config.patch({"inplace_buffers": False}):
        run_graph_test(
            model,
            [input],
            comparison_config=ComparisonConfig(),
            framework=Framework.TORCH,
            torch_options=options,
        )


@pytest.mark.single_device
@pytest.mark.parametrize("k", [1, 5, 16])
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("largest", [True, False])
def test_composite_topk_indices(k, dim, largest):
    """Test single-output composite that returns only the indices tensor."""
    if dim == 0 and k == 16:
        pytest.skip()

    class TopkIndicesModel(torch.nn.Module):
        def forward(self, x):
            return composite_topk_indices(x, k, dim, largest)

    options = {"tt_enable_composite_ops": False}

    input = torch.randn(8, 16)
    model = TopkIndicesModel()

    with torch._inductor.config.patch({"inplace_buffers": False}):
        run_graph_test(
            model,
            [input],
            comparison_config=ComparisonConfig(),
            framework=Framework.TORCH,
            torch_options=options,
        )


@pytest.mark.single_device
@pytest.mark.parametrize("k", [1, 5, 16])
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("largest", [True, False])
def test_patched_topk_values_only(k, dim, largest):
    """Patched torch.topk where only values are consumed.

    Exercises the workaround for the upstream XLA bug where a multi-output
    composite with a dead (unused) marked output crashes in _xla_warm_up_cache.
    The pass in handle_composite_ops detects that only values are consumed and
    routes to composite_topk_values (single output at pos=0), avoiding the crash.
    """
    if dim == 0 and k == 16:
        pytest.skip()

    class TopkValuesOnlyModel(torch.nn.Module):
        def forward(self, x):
            return torch.topk(x, k, dim, largest)[0]

    options = {"tt_enable_composite_ops": True}

    input = torch.randn(8, 16)
    model = TopkValuesOnlyModel()

    run_graph_test(
        model,
        [input],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        torch_options=options,
    )


@pytest.mark.single_device
@pytest.mark.parametrize("k", [1, 5, 16])
@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("largest", [True, False])
def test_patched_topk_indices_only(k, dim, largest):
    """Patched torch.topk where only indices are consumed.

    Exercises the workaround for the upstream XLA bug where a multi-output
    composite with a dead (unused) marked output crashes in _xla_warm_up_cache.
    The pass in handle_composite_ops detects that only indices are consumed and
    routes to composite_topk_indices (single output at pos=0), avoiding the crash.
    """
    if dim == 0 and k == 16:
        pytest.skip()

    class TopkIndicesOnlyModel(torch.nn.Module):
        def forward(self, x):
            return torch.topk(x, k, dim, largest)[1]

    options = {"tt_enable_composite_ops": True}

    input = torch.randn(8, 16)
    model = TopkIndicesOnlyModel()

    run_graph_test(
        model,
        [input],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        torch_options=options,
    )


@pytest.mark.single_device
@pytest.mark.parametrize(
    "use_weight, use_bias", [(True, True), (True, False), (False, False)]
)
@pytest.mark.parametrize(
    "batch_size, seq_len, embedding_dim",
    [(1, 32, 32), (1, 197, 768), (1, 1024, 768)],
)
def test_composite_layer_norm(use_weight, use_bias, batch_size, seq_len, embedding_dim):

    class LayerNormModel(torch.nn.Module):
        def __init__(self, normalized_shape):
            super().__init__()
            self.normalized_shape = normalized_shape

        def forward(self, x, weight=None, bias=None):
            return composite_layer_norm(
                x, self.normalized_shape, weight, bias, eps=1e-5
            )

    options = {"tt_enable_composite_ops": False}

    input_tensor = torch.randn(batch_size, seq_len, embedding_dim, dtype=torch.bfloat16)
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
