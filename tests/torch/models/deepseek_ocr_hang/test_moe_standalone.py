# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone moe_infer op tests — NO model dependency.

Uses synthetic inputs matching DeepSeek OCR Layer 1 shapes:
  hidden_size=1280, expert_intermediate=896, num_experts=64,
  top_k=6, seq_len=913 → total_dispatched=5478

These tests run in seconds (no model loading) and isolate:
  test_token_sort_only     — scatter_ + sum + argsort + fancy index + graph break
  test_single_expert_mlp   — one DeepseekV2MLP (gate_proj → SiLU → up_proj → down_proj)
  test_token_sort_1_expert — token sort + 1 expert MLP + zeros for rest + cat
  test_token_sort_all_experts — token sort + 64 expert MLPs + cat
"""

import pytest
import torch
import torch.nn as nn
from infra import Framework, run_op_test


HIDDEN_SIZE = 1280
EXPERT_INTERMEDIATE = 896
NUM_EXPERTS = 64
TOP_K = 6
SEQ_LEN = 913
TOTAL_DISPATCHED = SEQ_LEN * TOP_K  # 5478


class ExpertMLP(nn.Module):
    """Standalone DeepseekV2MLP matching expert dimensions."""

    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN_SIZE, EXPERT_INTERMEDIATE, bias=False)
        self.up_proj = nn.Linear(HIDDEN_SIZE, EXPERT_INTERMEDIATE, bias=False)
        self.down_proj = nn.Linear(EXPERT_INTERMEDIATE, HIDDEN_SIZE, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Test modules
# ---------------------------------------------------------------------------

class TokenSortModule(nn.Module):
    """Token sorting infrastructure only (no expert MLPs).

    Ops: new_zeros → scatter_ → sum → argsort → fancy index → .cpu() graph break
    → cat zeros (no real computation).
    """

    def __init__(self):
        super().__init__()
        self.num_experts = NUM_EXPERTS

    def forward(self, x, topk_idx):
        cnts = topk_idx.new_zeros((topk_idx.shape[0], self.num_experts))
        cnts.scatter_(1, topk_idx, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_idx.view(-1).argsort()
        sorted_tokens = x[idxs // topk_idx.shape[1]]

        tokens_per_expert = tokens_per_expert.cpu().tolist()

        outputs = []
        start_idx = 0
        for num_tokens in tokens_per_expert:
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            outputs.append(torch.zeros_like(sorted_tokens[start_idx:end_idx]))
            start_idx = end_idx

        return torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)


class SingleExpertModule(nn.Module):
    """Just one expert MLP on a fixed-size input (no token sorting)."""

    def __init__(self):
        super().__init__()
        self.expert = ExpertMLP()

    def forward(self, x):
        return self.expert(x)


class TokenSortNExpertsModule(nn.Module):
    """Token sort + N real expert MLPs (rest zeros) + cat."""

    def __init__(self, max_experts):
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.max_experts = max_experts
        self.experts = nn.ModuleList([ExpertMLP() for _ in range(NUM_EXPERTS)])

    def forward(self, x, topk_idx):
        cnts = topk_idx.new_zeros((topk_idx.shape[0], self.num_experts))
        cnts.scatter_(1, topk_idx, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_idx.view(-1).argsort()
        sorted_tokens = x[idxs // topk_idx.shape[1]]

        tokens_per_expert = tokens_per_expert.cpu().tolist()

        outputs = []
        start_idx = 0
        experts_run = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            if experts_run < self.max_experts:
                expert_out = self.experts[i](tokens_for_this_expert)
            else:
                expert_out = torch.zeros_like(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx
            experts_run += 1

        return torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)


# ---------------------------------------------------------------------------
# Synthetic inputs (deterministic seed for reproducibility)
# ---------------------------------------------------------------------------

def _make_token_sort_inputs():
    torch.manual_seed(42)
    x = torch.randn(SEQ_LEN, HIDDEN_SIZE, dtype=torch.bfloat16)
    topk_idx = torch.randint(0, NUM_EXPERTS, (SEQ_LEN, TOP_K), dtype=torch.int64)
    return [x, topk_idx]


def _make_expert_inputs():
    torch.manual_seed(42)
    num_tokens = SEQ_LEN * TOP_K // NUM_EXPERTS  # ~85 tokens per expert
    x = torch.randn(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16)
    return [x]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def token_sort_model():
    model = TokenSortModule().eval().to(torch.bfloat16)
    return model, _make_token_sort_inputs()


@pytest.fixture(scope="module")
def single_expert_model():
    model = SingleExpertModule().eval().to(torch.bfloat16)
    return model, _make_expert_inputs()


@pytest.fixture(scope="module")
def token_sort_1_expert_model():
    model = TokenSortNExpertsModule(max_experts=1).eval().to(torch.bfloat16)
    return model, _make_token_sort_inputs()


@pytest.fixture(scope="module")
def token_sort_all_experts_model():
    model = TokenSortNExpertsModule(max_experts=64).eval().to(torch.bfloat16)
    return model, _make_token_sort_inputs()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_token_sort_only(token_sort_model):
    """
    Token sort infrastructure only: scatter_ → sum → argsort → fancy index
    → .cpu() graph break → cat zeros.
    No expert MLPs. If this hangs, token sort ops are the culprit.
    """
    model, inputs = token_sort_model
    run_op_test(model, inputs, framework=Framework.TORCH)


@pytest.mark.single_device
def test_single_expert_mlp(single_expert_model):
    """
    One expert MLP on ~85 tokens: Linear(1280→896) → SiLU → Linear(1280→896) → mul
    → Linear(896→1280).
    No token sorting. If this hangs, the expert MLP itself is the culprit.
    """
    model, inputs = single_expert_model
    run_op_test(model, inputs, framework=Framework.TORCH)


@pytest.mark.single_device
def test_token_sort_1_expert(token_sort_1_expert_model):
    """
    Token sort + 1 real expert MLP + 63 zeros + cat.
    Combined test — matches the full-pipeline test_experts_1 behavior.
    """
    model, inputs = token_sort_1_expert_model
    run_op_test(model, inputs, framework=Framework.TORCH)


@pytest.mark.single_device
def test_token_sort_all_experts(token_sort_all_experts_model):
    """
    Token sort + all 64 expert MLPs + cat.
    Same as test_moe_experts_only but with synthetic inputs.
    """
    model, inputs = token_sort_all_experts_model
    run_op_test(model, inputs, framework=Framework.TORCH)
