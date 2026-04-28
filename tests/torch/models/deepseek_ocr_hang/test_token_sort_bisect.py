# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Token sort op bisection — NO model dependency, synthetic inputs.

test_token_sort_only hangs. This file bisects the individual ops
inside the token sort infrastructure to find the exact hanging op.

Token sort ops (modeling_deepseekv2.py lines 361-365, 399):
  1. new_zeros(913, 64)                          — create count matrix
  2. scatter_(1, topk_idx, 1)                    — mark expert assignments
  3. sum(dim=0) → tokens_per_expert [64]         — count per expert
  4. topk_idx.view(-1).argsort() → idxs [5478]   — sort indices
  5. x[idxs // topk_idx.shape[1]] → sorted [5478, 1280]  — gather rows
  6. tokens_per_expert.cpu().tolist()             — graph break

Shapes: seq_len=913, top_k=6, num_experts=64, hidden=1280
"""

import pytest
import torch
import torch.nn as nn
from infra import Framework, run_op_test


SEQ_LEN = 913
TOP_K = 6
NUM_EXPERTS = 64
HIDDEN_SIZE = 1280


# ---------------------------------------------------------------------------
# Individual op modules
# ---------------------------------------------------------------------------

class ScatterSumModule(nn.Module):
    """Ops 1-3: new_zeros → scatter_ → sum.

    Returns tokens_per_expert [64] — small fixed shape.
    """

    def forward(self, x, topk_idx):
        cnts = topk_idx.new_zeros((topk_idx.shape[0], NUM_EXPERTS))
        cnts.scatter_(1, topk_idx, 1)
        tokens_per_expert = cnts.sum(dim=0)
        return tokens_per_expert


class ArgsortModule(nn.Module):
    """Op 4 only: argsort on flattened topk_idx.

    Returns sorted indices [5478].
    """

    def forward(self, x, topk_idx):
        idxs = topk_idx.view(-1).argsort()
        return idxs


class FancyIndexModule(nn.Module):
    """Ops 4-5: argsort + fancy indexing x[idxs // shape[1]].

    Returns sorted_tokens [5478, 1280].
    """

    def forward(self, x, topk_idx):
        idxs = topk_idx.view(-1).argsort()
        sorted_tokens = x[idxs // topk_idx.shape[1]]
        return sorted_tokens


class ScatterSumArgsortModule(nn.Module):
    """Ops 1-4: new_zeros → scatter_ → sum → argsort.

    Returns (tokens_per_expert, idxs) via cat to single tensor for comparison.
    """

    def forward(self, x, topk_idx):
        cnts = topk_idx.new_zeros((topk_idx.shape[0], NUM_EXPERTS))
        cnts.scatter_(1, topk_idx, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_idx.view(-1).argsort()
        return idxs


class AllTokenSortOpsModule(nn.Module):
    """Ops 1-5: scatter_ + sum + argsort + fancy index (no graph break).

    Returns sorted_tokens [5478, 1280].
    This tests all the TT-side ops without the .cpu() graph break.
    """

    def forward(self, x, topk_idx):
        cnts = topk_idx.new_zeros((topk_idx.shape[0], NUM_EXPERTS))
        cnts.scatter_(1, topk_idx, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_idx.view(-1).argsort()
        sorted_tokens = x[idxs // topk_idx.shape[1]]
        return sorted_tokens


class AllTokenSortWithGraphBreakModule(nn.Module):
    """Ops 1-6: all token sort ops + .cpu() graph break + cat zeros.

    Same as TokenSortModule from test_moe_standalone.py — expected to HANG.
    """

    def forward(self, x, topk_idx):
        cnts = topk_idx.new_zeros((topk_idx.shape[0], NUM_EXPERTS))
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


# ---------------------------------------------------------------------------
# Synthetic inputs
# ---------------------------------------------------------------------------

def _make_inputs():
    torch.manual_seed(42)
    x = torch.randn(SEQ_LEN, HIDDEN_SIZE, dtype=torch.bfloat16)
    topk_idx = torch.randint(0, NUM_EXPERTS, (SEQ_LEN, TOP_K), dtype=torch.int64)
    return [x, topk_idx]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def inputs():
    return _make_inputs()


@pytest.fixture(scope="module")
def scatter_sum_model(inputs):
    return ScatterSumModule().eval(), inputs


@pytest.fixture(scope="module")
def argsort_model(inputs):
    return ArgsortModule().eval(), inputs


@pytest.fixture(scope="module")
def fancy_index_model(inputs):
    return FancyIndexModule().eval(), inputs


@pytest.fixture(scope="module")
def scatter_sum_argsort_model(inputs):
    return ScatterSumArgsortModule().eval(), inputs


@pytest.fixture(scope="module")
def all_ops_no_break_model(inputs):
    return AllTokenSortOpsModule().eval(), inputs


@pytest.fixture(scope="module")
def all_ops_with_break_model(inputs):
    return AllTokenSortWithGraphBreakModule().eval(), inputs


# ---------------------------------------------------------------------------
# Tests — run in order, each adds ops
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_scatter_sum(scatter_sum_model):
    """
    new_zeros → scatter_ → sum → returns [64].
    Tests scatter_ and sum on TT.
    """
    model, inputs = scatter_sum_model
    run_op_test(model, inputs, framework=Framework.TORCH)


@pytest.mark.single_device
def test_argsort(argsort_model):
    """
    view(-1) → argsort → returns [5478].
    Tests argsort on TT.
    """
    model, inputs = argsort_model
    run_op_test(model, inputs, framework=Framework.TORCH)


@pytest.mark.single_device
def test_fancy_index(fancy_index_model):
    """
    argsort + x[idxs // shape[1]] → returns [5478, 1280].
    Tests integer division + fancy indexing (gather) on TT.
    """
    model, inputs = fancy_index_model
    run_op_test(model, inputs, framework=Framework.TORCH)


@pytest.mark.single_device
def test_scatter_sum_argsort(scatter_sum_argsort_model):
    """
    scatter_ + sum + argsort combined → returns idxs [5478].
    Tests all counting + sorting ops together.
    """
    model, inputs = scatter_sum_argsort_model
    run_op_test(model, inputs, framework=Framework.TORCH)


@pytest.mark.single_device
def test_all_ops_no_graph_break(all_ops_no_break_model):
    """
    scatter_ + sum + argsort + fancy index → returns sorted_tokens [5478, 1280].
    All token sort ops WITHOUT the .cpu() graph break.
    If this passes but test_all_ops_with_graph_break hangs → graph break is the issue.
    """
    model, inputs = all_ops_no_break_model
    run_op_test(model, inputs, framework=Framework.TORCH)


@pytest.mark.single_device
def test_all_ops_with_graph_break(all_ops_with_break_model):
    """
    scatter_ + sum + argsort + fancy index + .cpu() graph break + cat zeros.
    Same as test_token_sort_only — expected to HANG.
    """
    model, inputs = all_ops_with_break_model
    run_op_test(model, inputs, framework=Framework.TORCH)
