# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Sanity tests for torch.gather on TT device.

Motivated by PCC drop (0.08) observed in the DeepSeek OCR masked_scatter
decomposition, where torch.gather(source, 0, idx_2d) is the sole op
responsible for the accuracy loss.

All tests use bfloat16 to match the model's runtime dtype.

Tests cover:
  1. DeepSeek OCR exact config: source [903, 1280] bfloat16, idx [913, 1280]
     int64, dim=0. This is the exact shape/dtype the model produces.
  2. Smaller shapes to verify gather works correctly in general.
  3. Dim=1 variant to check if the issue is dim-specific.
  4. Uniform vs random index patterns.

Related issues:
  - https://github.com/tenstorrent/tt-xla/issues/3316
  - https://github.com/tenstorrent/tt-xla/issues/3412
"""

import pytest
import torch
import torch.nn as nn
from infra import Framework, run_op_test


# ---------------------------------------------------------------------------
# DeepSeek OCR dimensions (confirmed via debug print on real model forward):
#   inputs_embeds[0]    shape=[913, 1280]  dtype=bfloat16
#   images_seq_mask[0]  shape=[913]        dtype=bool  num_true=903
#   images_in_this_batch shape=[903, 1280] dtype=bfloat16
# ---------------------------------------------------------------------------
S = 913
D = 1280
N = 903


# ---------------------------------------------------------------------------
# nn.Module wrappers
# ---------------------------------------------------------------------------

class GatherDim0(nn.Module):
    """torch.gather(input, 0, index) — gather rows."""

    def forward(self, input, index):
        return torch.gather(input, 0, index)


class GatherDim1(nn.Module):
    """torch.gather(input, 1, index) — gather columns."""

    def forward(self, input, index):
        return torch.gather(input, 1, index)


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------

def _build_deepseek_ocr_gather_inputs(seed=42):
    """Build gather inputs matching exactly what the DeepSeek OCR new
    masked_scatter decomposition produces.

    The decomposition does:
        mask_i       = mask_1d.long()                           # [913] int64
        source_idx   = cumsum(mask_i, 0) - 1                   # [913] int64
        source_idx   = clamp(source_idx, 0, N-1)               # [913] int64, values in [0, 902]
        source_idx_2d = source_idx.unsqueeze(-1).expand(S, D)  # [913, 1280] int64
        gathered     = torch.gather(source, 0, source_idx_2d)  # [913, 1280] bfloat16

    This function reproduces that exact pipeline on CPU to generate the
    gather inputs (source and source_idx_2d).
    """
    torch.manual_seed(seed)

    source = torch.randn(N, D, dtype=torch.bfloat16)

    mask_1d = torch.zeros(S, dtype=torch.bool)
    mask_1d[:N] = True
    mask_1d = mask_1d[torch.randperm(S)]

    mask_i = mask_1d.long()
    source_idx = torch.cumsum(mask_i, 0) - 1
    source_idx = torch.clamp(source_idx, 0, N - 1)
    source_idx_2d = source_idx.unsqueeze(-1).expand(S, D)

    return source, source_idx_2d


def _build_small_gather_inputs(seed=42):
    """Small gather inputs for quick sanity check: source [4, 8], idx [6, 8]."""
    torch.manual_seed(seed)
    source = torch.randn(4, 8, dtype=torch.bfloat16)
    index = torch.randint(0, 4, (6, 8), dtype=torch.long)
    return source, index


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.single_device
def test_gather_dim0_deepseek_ocr_bf16():
    """torch.gather dim=0 with DeepSeek OCR exact shapes and bfloat16.

    This matches the model's runtime config:
      - source: [903, 1280] bfloat16  (image features from vision encoder)
      - index:  [913, 1280] int64     (from cumsum on images_seq_mask)
      - dim=0

    Expected: PCC ~0.06 (known TT gather accuracy issue).
    """
    source, idx_2d = _build_deepseek_ocr_gather_inputs()
    model = GatherDim0()
    run_op_test(model, [source, idx_2d], framework=Framework.TORCH)


@pytest.mark.single_device
def test_gather_dim0_small_bf16():
    """torch.gather dim=0 with small shapes [4,8] bfloat16.

    If this passes, the gather op works for small tensors and the issue
    is specific to large shapes.
    """
    source, index = _build_small_gather_inputs()
    model = GatherDim0()
    run_op_test(model, [source, index], framework=Framework.TORCH)


@pytest.mark.single_device
def test_gather_dim1_deepseek_ocr_bf16():
    """torch.gather dim=1 with DeepSeek OCR-scale shapes, bfloat16.

    source: [913, 1280], index: [913, 903] — transposed usage.
    If dim=0 fails but dim=1 passes, the issue is dim-specific in TT.
    """
    torch.manual_seed(42)
    source = torch.randn(S, D, dtype=torch.bfloat16)
    index = torch.randint(0, D, (S, N), dtype=torch.long)
    model = GatherDim1()
    run_op_test(model, [source, index], framework=Framework.TORCH)


@pytest.mark.single_device
def test_gather_dim0_uniform_index_bf16():
    """torch.gather dim=0 with uniform (broadcast) index pattern.

    This matches the exact access pattern in the masked_scatter decomp:
    each row of source_idx_2d has the same value across all columns
    (because the index is unsqueeze + expand from a 1-D vector).

    source: [903, 1280] bfloat16, index: [913, 1280] int64.
    Index values: each row is constant (same value repeated D times).
    """
    torch.manual_seed(42)
    source = torch.randn(N, D, dtype=torch.bfloat16)
    row_indices = torch.randint(0, N, (S,), dtype=torch.long)
    index = row_indices.unsqueeze(-1).expand(S, D)
    model = GatherDim0()
    run_op_test(model, [source, index], framework=Framework.TORCH)


@pytest.mark.single_device
def test_gather_dim0_random_index_bf16():
    """torch.gather dim=0 with fully random index pattern.

    source: [903, 1280] bfloat16, index: [913, 1280] int64.
    Index values: fully random per element (not row-uniform).
    Compares with the uniform-index test to check if the index pattern
    matters.
    """
    torch.manual_seed(42)
    source = torch.randn(N, D, dtype=torch.bfloat16)
    index = torch.randint(0, N, (S, D), dtype=torch.long)
    model = GatherDim0()
    run_op_test(model, [source, index], framework=Framework.TORCH)
