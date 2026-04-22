# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Per-op isolation tests for the v2 masked_scatter decomposition (mul+add).

The v2 decomposition replaces torch.gather with mul+add index linearization:
    mask_i       = mask_1d.long()                                    # op1: cast
    source_idx   = torch.cumsum(mask_i, 0) - 1                      # op2: cumsum, op3: sub
    source_idx   = torch.clamp(source_idx, 0, max)                  # op4: clamp
    flat_source  = source.reshape(-1)                                # op5: flatten source
    col_idx      = torch.arange(D)                                   # op6: arange
    flat_idx     = source_idx.unsqueeze(-1) * D + col_idx            # op7: mul+add (replaces gather)
    gathered     = flat_source[flat_idx.reshape(-1)].reshape(S, D)   # op8: flat indexing
    result       = torch.where(mask.unsqueeze(-1), gathered, row)    # op9: where

Compared to the original decomposition, op5-op8 replace the unsqueeze+expand+gather
sequence, avoiding the ttnn.matmul precision bug on Wormhole.

Inputs match DeepSeek OCR real model forward:
  inputs_embeds[0]    shape=[913, 1280]  dtype=bfloat16
  images_seq_mask[0]  shape=[913]        dtype=bool  num_true=903
  images_in_this_batch shape=[903, 1280] dtype=bfloat16
"""

import pytest
import torch
import torch.nn as nn
from infra import Framework, run_op_test


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
# Op wrappers — each wraps one step of the v2 decomposition
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


class Op5FlattenSource(nn.Module):
    """source.reshape(-1) — flatten source for flat indexing."""
    def forward(self, source):
        return source.reshape(-1)


class Op7MulAdd(nn.Module):
    """flat_idx = source_idx.unsqueeze(-1) * D + col_idx.unsqueeze(0)

    The mul+add replacement for gather's index linearization.
    """
    def __init__(self, D):
        super().__init__()
        self.D = D

    def forward(self, source_idx):
        col_idx = torch.arange(self.D, device=source_idx.device, dtype=source_idx.dtype)
        return source_idx.unsqueeze(-1) * self.D + col_idx.unsqueeze(0)


class Op8FlatIndex(nn.Module):
    """gathered = flat_source[flat_idx.reshape(-1)].reshape(S, D)"""
    def __init__(self, S, D):
        super().__init__()
        self.S = S
        self.D = D

    def forward(self, flat_source, flat_idx):
        return flat_source[flat_idx.reshape(-1)].reshape(self.S, self.D)


class Op9Where(nn.Module):
    """torch.where(mask_2d, gathered_rows, inputs_embeds_row)"""
    def forward(self, mask_1d, gathered_rows, inputs_embeds_row):
        return torch.where(mask_1d.unsqueeze(-1), gathered_rows, inputs_embeds_row)


class Op2And3CumsumSub(nn.Module):
    """torch.cumsum(mask_i, 0) - 1 (combined)"""
    def forward(self, mask_i):
        return torch.cumsum(mask_i, 0) - 1


class Op1To4Pipeline(nn.Module):
    """Op1-Op4: cast + cumsum + sub + clamp.
    Returns source_idx — the per-row index into source."""
    def __init__(self, max_val):
        super().__init__()
        self.max_val = max_val

    def forward(self, mask_1d):
        mask_i = mask_1d.long()
        source_idx = torch.cumsum(mask_i, 0) - 1
        source_idx = torch.clamp(source_idx, 0, self.max_val)
        return source_idx


class Op1To8Pipeline(nn.Module):
    """Op1-Op8: cast + cumsum + sub + clamp + mul+add + flat_index.
    Returns gathered rows — the full index computation + lookup."""
    def __init__(self, max_val, D):
        super().__init__()
        self.max_val = max_val
        self.D = D

    def forward(self, mask_1d, source):
        S = mask_1d.shape[0]
        mask_i = mask_1d.long()
        source_idx = torch.cumsum(mask_i, 0) - 1
        source_idx = torch.clamp(source_idx, 0, self.max_val)
        flat_source = source.reshape(-1)
        col_idx = torch.arange(self.D, device=source.device, dtype=source_idx.dtype)
        flat_idx = source_idx.unsqueeze(-1) * self.D + col_idx.unsqueeze(0)
        return flat_source[flat_idx.reshape(-1)].reshape(S, self.D)


class FullNewDecompV2(nn.Module):
    """Complete v2 decomposition end-to-end (mul+add, no gather)."""
    def __init__(self, max_val, D):
        super().__init__()
        self.max_val = max_val
        self.D = D

    def forward(self, inputs_embeds_row, mask_1d, source):
        S = inputs_embeds_row.shape[0]
        mask_i = mask_1d.long()
        source_idx = torch.cumsum(mask_i, 0) - 1
        source_idx = torch.clamp(source_idx, 0, self.max_val)
        flat_source = source.reshape(-1)
        col_idx = torch.arange(self.D, device=source.device, dtype=source_idx.dtype)
        flat_idx = source_idx.unsqueeze(-1) * self.D + col_idx.unsqueeze(0)
        gathered_rows = flat_source[flat_idx.reshape(-1)].reshape(S, self.D)
        return torch.where(mask_1d.unsqueeze(-1), gathered_rows, inputs_embeds_row)


class FullNewDecompOriginal(nn.Module):
    """Original new decomposition (gather-based, for comparison)."""
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
    flat_source = source.reshape(-1)
    col_idx = torch.arange(D, dtype=clamped.dtype)
    flat_idx = clamped.unsqueeze(-1) * D + col_idx.unsqueeze(0)
    gathered = flat_source[flat_idx.reshape(-1)].reshape(S, D)
    return {
        "inputs_embeds_row": inputs_embeds_row,
        "mask_1d": mask_1d,
        "source": source,
        "mask_i": mask_i,
        "cumsum_result": cumsum_result,
        "sub_result": sub_result,
        "max_val": max_val,
        "clamped": clamped,
        "flat_source": flat_source,
        "col_idx": col_idx,
        "flat_idx": flat_idx,
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
def test_op5_flatten_source(intermediates):
    """source.reshape(-1) — flatten source for flat indexing."""
    m = Op5FlattenSource()
    run_op_test(m, [intermediates["source"]], framework=Framework.TORCH)


@pytest.mark.single_device
def test_op7_mul_add(intermediates):
    """flat_idx = source_idx * D + col_arange — mul+add index linearization.

    This is the key replacement for gather's index computation.
    """
    m = Op7MulAdd(D)
    run_op_test(m, [intermediates["clamped"]], framework=Framework.TORCH)


@pytest.mark.single_device
def test_op8_flat_index(intermediates):
    """flat_source[flat_idx] — flat lookup using computed indices."""
    m = Op8FlatIndex(S, D)
    run_op_test(
        m,
        [intermediates["flat_source"], intermediates["flat_idx"]],
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
def test_op9_where(intermediates):
    """torch.where(mask, gathered, original) — conditional select."""
    m = Op9Where()
    run_op_test(
        m,
        [intermediates["mask_1d"], intermediates["gathered"],
         intermediates["inputs_embeds_row"]],
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
def test_op1_to_op4_pipeline(intermediates):
    """Op1-Op4 combined: cast + cumsum + sub + clamp.
    Everything before the mul+add index step."""
    m = Op1To4Pipeline(intermediates["max_val"])
    run_op_test(
        m,
        [intermediates["mask_1d"]],
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
def test_op1_to_op8_pipeline(intermediates):
    """Op1-Op8 combined: cast + cumsum + sub + clamp + mul+add + flat_index.
    Full index computation + lookup — should PASS with mul+add fix."""
    m = Op1To8Pipeline(intermediates["max_val"], D)
    run_op_test(
        m,
        [intermediates["mask_1d"], intermediates["source"]],
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
def test_full_v2_decomp(intermediates):
    """Full v2 decomposition end-to-end (mul+add, no gather).
    Expected PCC ~1.0 — the matmul-free path."""
    m = FullNewDecompV2(intermediates["max_val"], D)
    run_op_test(
        m,
        [intermediates["inputs_embeds_row"], intermediates["mask_1d"],
         intermediates["source"]],
        framework=Framework.TORCH,
    )


@pytest.mark.single_device
def test_full_original_decomp(intermediates):
    """Full original decomposition (gather-based) for comparison.
    Expected PCC ~0.41 due to ttnn.matmul bug."""
    m = FullNewDecompOriginal(intermediates["max_val"])
    run_op_test(
        m,
        [intermediates["inputs_embeds_row"], intermediates["mask_1d"],
         intermediates["source"]],
        framework=Framework.TORCH,
    )
