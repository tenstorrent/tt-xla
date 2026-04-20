# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TT hardware sanity tests for masked_scatter_ decompositions in DeepSeek OCR.

Runs each variant (masked_scatter_, old decomp, new decomp) through
run_op_test which executes on both CPU and TT device and compares via PCC.

Context:
  - Issue #3316: masked_scatter_ fails with dynamic shapes on TT.
  - Issue #3412: old decomp runs cumsum on [S*D] int64 -> OOM on TT.
  - New decomp runs cumsum on [S] only -> fits in device memory.
  - PCC drops observed on TT device are due to TT op accuracy (cumsum,
    gather, where), not the decomposition itself — CPU-only tests prove
    bit-exact correctness for both decompositions.

Usage:
  pytest tests/torch/models/deepseek_ocr/test_masked_scatter_decomp_tt.py -svv
"""

import pytest
import torch
import torch.nn as nn
from infra import Framework, run_op_test
from tests.infra.evaluators.evaluation_config import ComparisonConfig, PccConfig


# ---------------------------------------------------------------------------
# Model-matching dimensions (confirmed via debug print on real model forward):
#   inputs_embeds[0]    shape=[913, 1280]  dtype=bfloat16
#   images_seq_mask[0]  shape=[913]        dtype=bool  num_true=903
#   images_in_this_batch shape=[903, 1280] dtype=bfloat16
# ---------------------------------------------------------------------------
S = 913
D = 1280
NUM_TRUE = 903


def _build_inputs(S, D, num_true, seed=42):
    torch.manual_seed(seed)
    inputs_embeds = torch.randn(S, D, dtype=torch.bfloat16)
    source = torch.randn(num_true, D, dtype=torch.bfloat16)

    mask_1d = torch.zeros(S, dtype=torch.bool)
    true_positions = torch.randperm(S)[:num_true].sort().values
    mask_1d[true_positions] = True
    return inputs_embeds, mask_1d, source


# ---------------------------------------------------------------------------
# nn.Module wrappers for run_op_test
# ---------------------------------------------------------------------------
class MaskedScatterReference(nn.Module):
    """Wraps torch.Tensor.masked_scatter_ as an nn.Module."""

    def forward(self, inputs_embeds, mask_1d, source):
        result = inputs_embeds.clone()
        result.masked_scatter_(mask_1d.unsqueeze(-1), source)
        return result


class MaskedScatterOldDecomp(nn.Module):
    """Old decomposition: flatten to 1D, cumsum on [S*D] elements."""

    def forward(self, inputs_embeds, mask_1d, source):
        mask = mask_1d.unsqueeze(-1)
        mask_broad, data = torch.broadcast_tensors(mask, inputs_embeds)
        mask_flat = mask_broad.reshape(-1)
        data_flat = data.reshape(-1)
        source_flat = source.reshape(-1)
        mask_i = mask_flat.long()
        source_idx = torch.cumsum(mask_i, 0) - 1
        source_idx = torch.clamp(source_idx, 0, source_flat.shape[0] - 1)
        gathered = source_flat[source_idx]
        result_flat = torch.where(mask_flat, gathered, data_flat)
        return result_flat.view_as(inputs_embeds)


class MaskedScatterNewDecomp(nn.Module):
    """New decomposition: row-level cumsum on [S], 2D gather."""

    def forward(self, inputs_embeds, mask_1d, source):
        mask_i = mask_1d.long()
        source_idx = torch.cumsum(mask_i, 0) - 1
        source_idx = torch.clamp(source_idx, 0, source.shape[0] - 1)
        source_idx_2d = source_idx.unsqueeze(-1).expand_as(inputs_embeds)
        gathered_rows = torch.gather(source, 0, source_idx_2d)
        return torch.where(mask_1d.unsqueeze(-1), gathered_rows, inputs_embeds)


class MaskedScatterNewDecompCompilerDisabled(nn.Module):
    """New decomposition with torch.compiler.disable on forward.

    Prevents XLA from tracing/compiling the masked_scatter logic so it
    runs eagerly on CPU even when the rest of the graph is on TT device.
    """

    @torch.compiler.disable
    def forward(self, inputs_embeds, mask_1d, source):
        mask_i = mask_1d.long()
        source_idx = torch.cumsum(mask_i, 0) - 1
        source_idx = torch.clamp(source_idx, 0, source.shape[0] - 1)
        source_idx_2d = source_idx.unsqueeze(-1).expand_as(inputs_embeds)
        gathered_rows = torch.gather(source, 0, source_idx_2d)
        return torch.where(mask_1d.unsqueeze(-1), gathered_rows, inputs_embeds)


class MaskedScatterOldDecompCompilerDisabled(nn.Module):
    """Old decomposition with torch.compiler.disable on forward."""

    @torch.compiler.disable
    def forward(self, inputs_embeds, mask_1d, source):
        mask = mask_1d.unsqueeze(-1)
        mask_broad, data = torch.broadcast_tensors(mask, inputs_embeds)
        mask_flat = mask_broad.reshape(-1)
        data_flat = data.reshape(-1)
        source_flat = source.reshape(-1)
        mask_i = mask_flat.long()
        source_idx = torch.cumsum(mask_i, 0) - 1
        source_idx = torch.clamp(source_idx, 0, source_flat.shape[0] - 1)
        gathered = source_flat[source_idx]
        result_flat = torch.where(mask_flat, gathered, data_flat)
        return result_flat.view_as(inputs_embeds)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def model_inputs():
    inputs_embeds, mask_1d, source = _build_inputs(S, D, NUM_TRUE)
    return [inputs_embeds, mask_1d, source]


@pytest.fixture
def comparison_config():
    return ComparisonConfig(
        pcc=PccConfig(required_pcc=0.99),
    )


# ---------------------------------------------------------------------------
# Sanity 1: masked_scatter_ -- crashes on TT (repeat_interleave/transpose)
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_masked_scatter_reference_tt(model_inputs, comparison_config):
    """
    masked_scatter_ on TT device.
    Crashes with TT_FATAL in ttnn::repeat_interleave -> transpose_impl.
    """
    model = MaskedScatterReference()
    model.eval()

    run_op_test(
        model,
        model_inputs,
        comparison_config=comparison_config,
        framework=Framework.TORCH,
    )


# ---------------------------------------------------------------------------
# Sanity 2: new decomposition -- PCC drop is TT op accuracy issue
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_masked_scatter_new_decomp_tt(model_inputs, comparison_config):
    """
    New decomposition (row-level cumsum on [S] + 2D gather) on TT device.
    cumsum input: [913] int64 instead of [1168640] int64.
    Runs without crash but PCC drops due to TT op accuracy.
    """
    model = MaskedScatterNewDecomp()
    model.eval()

    run_op_test(
        model,
        model_inputs,
        comparison_config=comparison_config,
        framework=Framework.TORCH,
    )


# ---------------------------------------------------------------------------
# Sanity 3: old decomposition -- crashes on TT (repeat_interleave)
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_masked_scatter_old_decomp_tt(model_inputs, comparison_config):
    """
    Old decomposition (flatten to 1D, cumsum on [S*D]) on TT device.
    Crashes with TT_FATAL in ttnn::repeat_interleave -> transpose_impl.
    """
    model = MaskedScatterOldDecomp()
    model.eval()

    run_op_test(
        model,
        model_inputs,
        comparison_config=comparison_config,
        framework=Framework.TORCH,
    )


# ---------------------------------------------------------------------------
# Sanity 4: new decomposition with torch.compiler.disable
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_masked_scatter_new_decomp_compiler_disabled_tt(
    model_inputs, comparison_config
):
    """
    New decomposition with @torch.compiler.disable.
    Prevents XLA from tracing the op — runs eagerly on CPU, bypassing
    TT op accuracy issues. Should produce PCC ~1.0 since the logic
    never touches TT device.
    """
    model = MaskedScatterNewDecompCompilerDisabled()
    model.eval()

    run_op_test(
        model,
        model_inputs,
        comparison_config=comparison_config,
        framework=Framework.TORCH,
    )


# ---------------------------------------------------------------------------
# Sanity 5: old decomposition with torch.compiler.disable
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_masked_scatter_old_decomp_compiler_disabled_tt(
    model_inputs, comparison_config
):
    """
    Old decomposition with @torch.compiler.disable.
    Prevents XLA from tracing — runs eagerly on CPU, bypassing the
    repeat_interleave crash. Should produce PCC ~1.0.
    """
    model = MaskedScatterOldDecompCompilerDisabled()
    model.eval()

    run_op_test(
        model,
        model_inputs,
        comparison_config=comparison_config,
        framework=Framework.TORCH,
    )
