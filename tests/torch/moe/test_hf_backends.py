# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
import torch_xla
from tt_torch.attention_backend import tt_sdpa_attention_forward
from tt_torch.moe_backend import tt_experts_forward
from utils import Category


class DummyExperts(nn.Module):
    def __init__(self, num_experts, hidden_dim, intermediate_dim, is_transposed=False):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.is_transposed = is_transposed

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
    shlo_op_name="stablehlo.custom_call",
)
@pytest.mark.parametrize("is_transposed", [False, True])
def test_tt_experts_forward_dp(is_transposed):
    num_experts = 4
    hidden_dim = 64
    intermediate_dim = 128
    seq_len = 64  # Multiple of 32
    num_experts_per_tok = 2

    experts = DummyExperts(num_experts, hidden_dim, intermediate_dim, is_transposed)

    # Generate inputs
    hidden_states = torch.randn(seq_len, hidden_dim)
    top_k_index = torch.randint(0, num_experts, (seq_len, num_experts_per_tok))
    top_k_weights = torch.rand(seq_len, num_experts_per_tok)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

    # Reference eager forward
    # (Since we don't have HF's eager backend here, we just check that it runs and returns correct shape)
    # The actual correctness of sparse_matmul vs dense is tested in ops tests.
    # Here we test the HF interface integration.

    device = torch_xla.device()
    experts_tt = experts.to(device)
    hidden_states_tt = hidden_states.to(device)
    top_k_index_tt = top_k_index.to(device)
    top_k_weights_tt = top_k_weights.to(device)

    # Run on TT
    compiled_experts = torch.compile(tt_experts_forward, backend="tt")
    output_tt = compiled_experts(
        experts_tt, hidden_states_tt, top_k_index_tt, top_k_weights_tt
    )

    assert output_tt.shape == (seq_len, hidden_dim)
    assert output_tt.device == device


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.GRAPH_TEST,
    shlo_op_name="stablehlo.custom_call",
)
def test_tt_sdpa_attention_forward():
    batch_size = 2
    num_heads = 4
    seq_len = 64  # Multiple of 32
    head_dim = 64

    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim)

    device = torch_xla.device()
    query_tt = query.to(device)
    key_tt = key.to(device)
    value_tt = value.to(device)

    compiled_sdpa = torch.compile(tt_sdpa_attention_forward, backend="tt")
    # Note: tt_sdpa_attention_forward expects query, key, value as [B, num_heads, seq_len, head_dim]
    output_tt, _ = compiled_sdpa(
        None, query_tt, key_tt, value_tt, attention_mask=None, is_causal=True
    )

    # Output should be [B, seq_len, num_heads, head_dim]
    assert output_tt.shape == (batch_size, seq_len, num_heads, head_dim)
    assert output_tt.device == device
