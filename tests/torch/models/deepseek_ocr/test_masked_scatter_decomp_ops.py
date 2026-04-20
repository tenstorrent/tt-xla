# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Per-op isolation tests for the new masked_scatter decomposition.

The full decomposition (CPU vs TT) has PCC=0.41. These tests run each
individual op on TT vs CPU to find which one causes the PCC drop.

The new decomp:
    mask_i       = mask_1d.long()                                    # op1: cast
    source_idx   = torch.cumsum(mask_i, 0) - 1                      # op2: cumsum, op3: sub
    source_idx   = torch.clamp(source_idx, 0, max)                  # op4: clamp
    source_idx_2d = source_idx.unsqueeze(-1).expand_as(row)         # op5: unsqueeze+expand
    gathered     = torch.gather(source, 0, source_idx_2d)           # op6: gather
    result       = torch.where(mask.unsqueeze(-1), gathered, row)   # op7: where

Inputs match DeepSeek OCR real model forward (confirmed via debug print):
  inputs_embeds[0]    shape=[913, 1280]  dtype=bfloat16
  images_seq_mask[0]  shape=[913]        dtype=bool  num_true=903
  images_in_this_batch shape=[903, 1280] dtype=bfloat16
"""

import pytest
import torch
import torch.nn as nn
from infra import Framework, run_op_test


# ---------------------------------------------------------------------------
# Realistic input factory (matches DeepSeek OCR model forward)
# ---------------------------------------------------------------------------

S, D = 913, 1280
NUM_TRUE = 903


@pytest.fixture(scope="module")
def decomp_tensors():
    """Build realistic inputs for the decomposition, matching DeepSeek OCR."""
    torch.manual_seed(42)
    inputs_embeds_row = torch.randn(S, D, dtype=torch.bfloat16)
    mask_1d = torch.zeros(S, dtype=torch.bool)
    mask_1d[:NUM_TRUE] = True
    mask_1d = mask_1d[torch.randperm(S)]
    source = torch.randn(int(mask_1d.sum().item()), D, dtype=torch.bfloat16)
    return inputs_embeds_row, mask_1d, source


# ---------------------------------------------------------------------------
# Op wrappers — each wraps exactly one step of the decomposition
# ---------------------------------------------------------------------------

class Op1CastLong(nn.Module):
    """mask_1d.long()"""
    def forward(self, mask_1d):
        return mask_1d.long()


class Op2Cumsum(nn.Module):
    """torch.cumsum(mask_i, 0)"""
    def forward(self, mask_i):
        return torch.cumsum(mask_i, 0)


class Op3Sub(nn.Module):
    """cumsum_result - 1"""
    def forward(self, cumsum_result):
        return cumsum_result - 1


class Op4Clamp(nn.Module):
    """torch.clamp(source_idx, 0, max_val)"""
    def __init__(self, max_val):
        super().__init__()
        self.max_val = max_val

    def forward(self, source_idx):
        return torch.clamp(source_idx, 0, self.max_val)


class Op5UnsqueezeExpand(nn.Module):
    """source_idx.unsqueeze(-1).expand_as(inputs_embeds_row)"""
    def forward(self, source_idx, inputs_embeds_row):
        return source_idx.unsqueeze(-1).expand_as(inputs_embeds_row)


class Op6Gather(nn.Module):
    """torch.gather(source, 0, source_idx_2d)"""
    def forward(self, source, source_idx_2d):
        return torch.gather(source, 0, source_idx_2d)


class Op7Where(nn.Module):
    """torch.where(mask_2d, gathered_rows, inputs_embeds_row)"""
    def forward(self, mask_1d, gathered_rows, inputs_embeds_row):
        return torch.where(mask_1d.unsqueeze(-1), gathered_rows, inputs_embeds_row)


class Op2And3CumsumSub(nn.Module):
    """torch.cumsum(mask_i, 0) - 1  (combined)"""
    def forward(self, mask_i):
        return torch.cumsum(mask_i, 0) - 1


class Op1To5Pipeline(nn.Module):
    """Op1 through Op5: cast + cumsum + sub + clamp + unsqueeze/expand.
    Returns the 2-D index tensor that would be fed into gather."""
    def __init__(self, max_val):
        super().__init__()
        self.max_val = max_val

    def forward(self, mask_1d, inputs_embeds_row):
        mask_i = mask_1d.long()
        source_idx = torch.cumsum(mask_i, 0) - 1
        source_idx = torch.clamp(source_idx, 0, self.max_val)
        source_idx_2d = source_idx.unsqueeze(-1).expand_as(inputs_embeds_row)
        return source_idx_2d


class Op1To6Pipeline(nn.Module):
    """Op1 through Op6: cast + cumsum + sub + clamp + unsqueeze/expand + gather.
    Returns gathered rows — includes the broken gather to confirm it causes
    the PCC drop."""
    def __init__(self, max_val):
        super().__init__()
        self.max_val = max_val

    def forward(self, mask_1d, inputs_embeds_row, source):
        mask_i = mask_1d.long()
        source_idx = torch.cumsum(mask_i, 0) - 1
        source_idx = torch.clamp(source_idx, 0, self.max_val)
        source_idx_2d = source_idx.unsqueeze(-1).expand_as(inputs_embeds_row)
        gathered_rows = torch.gather(source, 0, source_idx_2d)
        return gathered_rows


class FullNewDecomp(nn.Module):
    """Complete new decomposition end-to-end (for baseline comparison)."""
    def __init__(self, max_val):
        super().__init__()
        self.max_val = max_val

    def forward(self, inputs_embeds_row, mask_1d, source):
        mask_i = mask_1d.long()
        source_idx = torch.cumsum(mask_i, 0) - 1
        source_idx = torch.clamp(source_idx, 0, self.max_val)
        source_idx_2d = source_idx.unsqueeze(-1).expand_as(inputs_embeds_row)
        gathered_rows = torch.gather(source, 0, source_idx_2d)
        return torch.where(mask_1d.unsqueeze(-1), gathered_rows, inputs_embeds_row)


# ---------------------------------------------------------------------------
# Precompute intermediate tensors for each op's input (on CPU)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def intermediates(decomp_tensors):
    inputs_embeds_row, mask_1d, source = decomp_tensors
    mask_i = mask_1d.long()
    cumsum_result = torch.cumsum(mask_i, 0)
    sub_result = cumsum_result - 1
    max_val = source.shape[0] - 1
    clamped = torch.clamp(sub_result, 0, max_val)
    idx_2d = clamped.unsqueeze(-1).expand_as(inputs_embeds_row)
    gathered = torch.gather(source, 0, idx_2d)
    return {
        "inputs_embeds_row": inputs_embeds_row,
        "mask_1d": mask_1d,
        "source": source,
        "mask_i": mask_i,
        "cumsum_result": cumsum_result,
        "sub_result": sub_result,
        "max_val": max_val,
        "clamped": clamped,
        "idx_2d": idx_2d,
        "gathered": gathered,
    }


# ---------------------------------------------------------------------------
# Tests — one per op, CPU vs TT
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_op1_cast_long(intermediates):
    """mask_1d.long() — bool to int64 cast."""
    m = Op1CastLong()
    run_op_test(m, [intermediates["mask_1d"]], framework=Framework.TORCH)


@pytest.mark.single_device
def test_op2_cumsum(intermediates):
    """torch.cumsum(mask_i, 0) — cumulative sum on int64 [913]."""
    m = Op2Cumsum()
    run_op_test(m, [intermediates["mask_i"]], framework=Framework.TORCH)


@pytest.mark.single_device
def test_op3_sub(intermediates):
    """cumsum_result - 1 — element-wise subtract."""
    m = Op3Sub()
    run_op_test(m, [intermediates["cumsum_result"]], framework=Framework.TORCH)


@pytest.mark.single_device
def test_op2_op3_cumsum_sub(intermediates):
    """torch.cumsum(mask_i, 0) - 1 — combined cumsum + subtract."""
    m = Op2And3CumsumSub()
    run_op_test(m, [intermediates["mask_i"]], framework=Framework.TORCH)


@pytest.mark.single_device
def test_op4_clamp(intermediates):
    """torch.clamp(source_idx, 0, max) — clamp indices."""
    m = Op4Clamp(intermediates["max_val"])
    run_op_test(m, [intermediates["sub_result"]], framework=Framework.TORCH)


@pytest.mark.single_device
def test_op5_unsqueeze_expand(intermediates):
    """source_idx.unsqueeze(-1).expand_as(row) — broadcast [913] -> [913, 1280]."""
    m = Op5UnsqueezeExpand()
    run_op_test(
        m,
        [intermediates["clamped"], intermediates["inputs_embeds_row"]],
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
def test_op6_gather(intermediates):
    """torch.gather(source, 0, idx_2d) — row-wise gather [913,1280] from source."""
    m = Op6Gather()
    run_op_test(
        m,
        [intermediates["source"], intermediates["idx_2d"]],
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
def test_op7_where(intermediates):
    """torch.where(mask, gathered, original) — conditional select."""
    m = Op7Where()
    run_op_test(
        m,
        [intermediates["mask_1d"], intermediates["gathered"],
         intermediates["inputs_embeds_row"]],
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
def test_op1_to_op5_pipeline(intermediates):
    """Op1-Op5 combined: cast + cumsum + sub + clamp + expand.
    Everything before gather — should PASS if gather is the only issue."""
    m = Op1To5Pipeline(intermediates["max_val"])
    run_op_test(
        m,
        [intermediates["mask_1d"], intermediates["inputs_embeds_row"]],
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
def test_op1_to_op6_pipeline(intermediates):
    """Op1-Op6 combined: cast + cumsum + sub + clamp + expand + gather.
    Adds gather on top of the passing pipeline — should FAIL with low PCC,
    confirming gather is the source of the drop."""
    m = Op1To6Pipeline(intermediates["max_val"])
    run_op_test(
        m,
        [intermediates["mask_1d"], intermediates["inputs_embeds_row"],
         intermediates["source"]],
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
def test_full_new_decomp(intermediates):
    """Full new decomposition end-to-end (baseline — expected PCC ~0.41)."""
    m = FullNewDecomp(intermediates["max_val"])
    run_op_test(
        m,
        [intermediates["inputs_embeds_row"], intermediates["mask_1d"],
         intermediates["source"]],
        framework=Framework.TORCH,
    )
