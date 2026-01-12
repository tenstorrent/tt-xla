# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra.utilities.types import Framework
from torch._dynamo import register_backend
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from tt_torch.backend.passes import run_fusion_passes

from tests.infra.comparators.comparison_config import ComparisonConfig
from tests.infra.testers.single_chip.graph.graph_tester import run_graph_test

# ============= Graph Capture Utilities =============

_captured_gm = None


@register_backend(name="capture_graph")
def capture_backend(gm, example_inputs, options=None):
    """Capture the graph that dynamo provides to backends."""
    global _captured_gm
    _captured_gm = gm
    return gm.forward


def get_dynamo_graph(model, example_inputs):
    """Get the GraphModule that dynamo would provide to a backend."""
    global _captured_gm
    _captured_gm = None

    # Reset dynamo to ensure fresh compilation
    torch._dynamo.reset()

    # Compile with capture backend
    compiled = torch.compile(model, backend="capture_graph")

    # Execute to trigger compilation
    compiled(*example_inputs)

    return _captured_gm


# ============= Graph Inspection Utilities =============


def find_nodes_by_target(gm, target):
    """Find all call_function nodes with the specified target."""
    return [
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target == target
    ]


def has_op_in_graph(gm, target):
    """Check if a specific operation exists in the graph."""
    return len(find_nodes_by_target(gm, target)) > 0


# ============= Graph-Level Fusion Tests =============


@pytest.mark.single_device
@pytest.mark.push
@pytest.mark.parametrize("hidden_size", [32, 768])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rms_norm_fusion_graph_level(hidden_size, dtype):
    """Test that RMS norm fusion creates rms_norm op in the dynamo graph."""
    # Setup
    model = LlamaRMSNorm(hidden_size).to(dtype)
    input_tensor = torch.randn(1, 32, hidden_size, dtype=dtype)

    # Get the dynamo graph (same as what backend receives)
    gm = get_dynamo_graph(model, [input_tensor])

    # Verify rms_norm NOT present before fusion
    assert not has_op_in_graph(
        gm, torch.nn.functional.rms_norm
    ), "rms_norm should not be in graph before fusion"

    # Apply fusion passes (same function used by the backend)
    gm = run_fusion_passes(gm)

    # Verify rms_norm IS present after fusion
    assert has_op_in_graph(
        gm, torch.nn.functional.rms_norm
    ), "torch.nn.functional.rms_norm should be in graph after fusion"


# ============= End-to-End Fusion Tests =============


@pytest.mark.single_device
@pytest.mark.push
@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size",
    [(1, 32, 32), (1, 128, 768), (1, 1024, 768)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_llama_rms_norm_fusion(batch_size, seq_len, hidden_size, dtype):

    options = {
        "tt_enable_fusion_passes": True,
        "tt_enable_composite_ops": True,
    }

    model = LlamaRMSNorm(hidden_size)
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)

    run_graph_test(
        model,
        [input_tensor],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        torch_options=options,
    )
