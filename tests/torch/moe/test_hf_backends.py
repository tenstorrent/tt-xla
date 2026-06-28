# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra import ComparisonConfig
from infra.evaluators.torch_comparison_evaluator import TorchComparisonEvaluator
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from tt_torch.moe_backend import register_tt_moe_backend, tt_experts_forward
from utils import Category


class DummyExperts(nn.Module):
    def __init__(self, num_experts, hidden_dim, intermediate_dim, is_transposed=False):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.is_transposed = is_transposed
        # Attrs read by HF `batched_mm` fallback when called on CPU tensors.
        self.has_gate = True
        self.has_bias = False

        if is_transposed:
            self.gate_up_proj = nn.Parameter(
                torch.randn(num_experts, hidden_dim, 2 * intermediate_dim)
            )
            self.down_proj = nn.Parameter(
                torch.randn(num_experts, intermediate_dim, hidden_dim)
            )
        else:
            self.gate_up_proj = nn.Parameter(
                torch.randn(num_experts, 2 * intermediate_dim, hidden_dim)
            )
            self.down_proj = nn.Parameter(
                torch.randn(num_experts, hidden_dim, intermediate_dim)
            )

    def _apply_gate(self, gate_up_out):
        # Simple GLU for testing
        gate, up = gate_up_out.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.GRAPH_TEST,
)
@pytest.mark.parametrize("is_transposed", [False, True])
def test_tt_experts_forward_cpu_fallback(is_transposed):
    """CPU tensors should fall back to HF `batched_mm` instead of hitting the
    multi-chip SPMD path."""
    xr.set_device_type("TT")
    num_experts = 4
    hidden_dim = 64
    intermediate_dim = 128
    seq_len = 64  # Multiple of 32
    num_experts_per_tok = 2

    experts = DummyExperts(num_experts, hidden_dim, intermediate_dim, is_transposed)

    hidden_states = torch.randn(seq_len, hidden_dim)
    top_k_index = torch.randint(0, num_experts, (seq_len, num_experts_per_tok))
    top_k_weights = torch.rand(seq_len, num_experts_per_tok)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

    out = tt_experts_forward(experts, hidden_states, top_k_index, top_k_weights)
    assert out.shape == (seq_len, hidden_dim)
    assert out.device.type == "cpu"


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.skip(reason="DiffusionGemma needs transformers==5.12.0")
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_diffusion_gemma_experts_grouped_cpu_vs_tt_moe():
    """DiffusionGemma (26B-A4B) experts: grouped_mm (CPU golden) vs the multi-chip
    expert-parallel tt_moe path. Experts are sharded on the expert dim; tt_moe reads
    the global SPMD mesh to pick the EP cluster axis."""
    # lazy import
    from transformers.models.diffusion_gemma.configuration_diffusion_gemma import (
        DiffusionGemmaTextConfig,
    )
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaTextExperts,
    )

    torch.manual_seed(0)
    register_tt_moe_backend()

    # Real MoE-expert dims: fused gate_up, GELUTanh gate.
    config = DiffusionGemmaTextConfig(
        hidden_size=2816,
        moe_intermediate_size=704,
        num_experts=128,
        top_k_experts=8,
        hidden_activation="gelu_pytorch_tanh",
    )
    experts = DiffusionGemmaTextExperts(config)
    with torch.no_grad():
        for p in experts.parameters():
            p.normal_(0, 0.02)  # gate_up_proj/down_proj are torch.empty -> initialize
    experts = experts.to(torch.bfloat16).eval()
    experts.config = config

    # randn inputs with the model's shapes/dtypes. topk -> distinct experts per token
    # (real top-k never repeats one); weights are normalized like the router output.
    seq_len = 32
    hidden_states = torch.randn(seq_len, config.hidden_size, dtype=torch.bfloat16)
    top_k_index = (
        torch.randn(seq_len, config.num_experts)
        .topk(config.top_k_experts, dim=-1)
        .indices
    )
    top_k_weights = torch.rand(seq_len, config.top_k_experts)
    top_k_weights = (top_k_weights / top_k_weights.sum(-1, keepdim=True)).to(
        torch.float32
    )

    # CPU golden -- grouped_mm
    config._experts_implementation = "grouped_mm"
    with torch.no_grad():
        golden = experts(hidden_states, top_k_index, top_k_weights)

    # TT run -- tt_moe (expert-parallel: sharded experts + global SPMD mesh)
    config._experts_implementation = "tt_moe"
    xr.set_device_type("TT")
    enable_spmd()
    device = xm.xla_device()
    experts = experts.to(device)

    num_devices = xr.global_runtime_device_count()
    mesh = get_mesh((1, num_devices), ("batch", "model"))
    xs.set_global_mesh(mesh)
    xs.mark_sharding(experts.gate_up_proj, mesh, (("batch", "model"), None, None))
    xs.mark_sharding(experts.down_proj, mesh, (("batch", "model"), None, None))

    compiled = torch.compile(experts, backend="tt")
    with torch.no_grad():
        tt_out = compiled(
            hidden_states.to(device),
            top_k_index.to(device),
            top_k_weights.to(device),
        ).to("cpu")

    TorchComparisonEvaluator(ComparisonConfig()).evaluate(tt_out, golden)
