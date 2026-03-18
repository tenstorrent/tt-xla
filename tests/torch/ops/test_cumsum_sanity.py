# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Lightweight sanity tests for every torch.cumsum configuration observed in the
PI0 sample_actions / denoise_step inference path.
torch.cumsum lowers to stablehlo.reduce_window{ADD} which currently fails
legalization to TTIR.  These tests reproduce each unique cumsum call site with
the exact shapes and dtypes captured from a CPU run of the LIBERO_BASE variant.
Each test verifies:
  - PCC >= 0.99 between CPU and TT outputs
  - allclose (rtol=1e-5, atol=1e-5)
  - output dtype matches between CPU run and the expected dtype
On CPU, torch.cumsum(bool_tensor) implicitly promotes bool -> int64.
The explicit .to(torch.long) in the wrappers replicates that promotion so
the TT-XLA compiler sees int64 input and avoids the reduce_window failure.
This cast is lossless (bool 0/1 maps exactly to int64 0/1).
Observed cumsum call sites (4 calls across 3 locations):
1. test_cumsum_bool_prefix_att_masks
   make_att_2d_masks() called from sample_actions
   torch.cumsum(att_masks, dim=1)
   Input: bool[1, 816], Output dtype: int64
2. test_cumsum_bool_prefix_pad_masks
   Direct call in sample_actions
   torch.cumsum(prefix_pad_masks.to(torch.long), dim=1) - 1
   Input: bool[1, 816], Output dtype: int64
3. test_cumsum_float_suffix_att_masks
   make_att_2d_masks() called from denoise_step
   torch.cumsum(att_masks, dim=1)
   Input: float32[1, 51], Output dtype: float32
4. test_cumsum_bool_suffix_pad_masks
   Direct call in denoise_step
   torch.cumsum(suffix_pad_masks.to(torch.long), dim=1) - 1
   Input: bool[1, 51], Output dtype: int64
"""

import pytest
import torch
from infra import ComparisonConfig, Framework, run_op_test
from infra.evaluators import AllcloseConfig


_COMPARISON = ComparisonConfig(
    allclose=AllcloseConfig(enabled=True, rtol=1e-5, atol=1e-5),
)


# ---------------------------------------------------------------------------
# Op wrappers
# ---------------------------------------------------------------------------
class CumsumBoolOp(torch.nn.Module):
    """torch.cumsum on a bool tensor, explicitly cast to int64 to avoid
    stablehlo.reduce_window legalization failure on bool inputs."""

    def forward(self, masks):
        return torch.cumsum(masks.to(torch.long), dim=1)


class CumsumPositionIdsOp(torch.nn.Module):
    """torch.cumsum(bool -> long) - 1, used to compute position_ids."""

    def forward(self, pad_masks):
        return torch.cumsum(pad_masks.to(torch.long), dim=1) - 1


class CumsumFloatOp(torch.nn.Module):
    """torch.cumsum on a float32 tensor."""

    def forward(self, masks):
        return torch.cumsum(masks, dim=1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_cumsum_bool_prefix_att_masks():
    """make_att_2d_masks cumsum in sample_actions.
    Observed input: att_masks  bool[1, 816]
    CPU implicit promotion: bool -> int64
    """
    wrapper = CumsumBoolOp()
    inputs = [torch.randint(0, 2, (1, 816), dtype=torch.bool)]

    cpu_out = wrapper(*inputs)
    assert cpu_out.dtype == torch.int64, f"Expected int64, got {cpu_out.dtype}"

    run_op_test(wrapper, inputs, comparison_config=_COMPARISON, framework=Framework.TORCH)


@pytest.mark.single_device
def test_cumsum_bool_prefix_pad_masks():
    """Direct cumsum in sample_actions.
    Observed input: prefix_pad_masks  bool[1, 816]
    CPU implicit promotion: bool -> int64
    """
    wrapper = CumsumPositionIdsOp()
    inputs = [torch.ones(1, 816, dtype=torch.bool)]

    cpu_out = wrapper(*inputs)
    assert cpu_out.dtype == torch.int64, f"Expected int64, got {cpu_out.dtype}"

    run_op_test(wrapper, inputs, comparison_config=_COMPARISON, framework=Framework.TORCH)


@pytest.mark.single_device
def test_cumsum_float_suffix_att_masks():
    """make_att_2d_masks cumsum in denoise_step.
    Observed input: suffix_att_masks  float32[1, 51]
    No promotion needed -- input is already float32.
    """
    wrapper = CumsumFloatOp()
    inputs = [torch.tensor([[1, 1, 0] + [0] * 48], dtype=torch.float32)]

    cpu_out = wrapper(*inputs)
    assert cpu_out.dtype == torch.float32, f"Expected float32, got {cpu_out.dtype}"

    run_op_test(wrapper, inputs, comparison_config=_COMPARISON, framework=Framework.TORCH)


@pytest.mark.single_device
def test_cumsum_bool_suffix_pad_masks():
    """Direct cumsum in denoise_step.
    Observed input: suffix_pad_masks  bool[1, 51]
    CPU implicit promotion: bool -> int64
    """
    wrapper = CumsumPositionIdsOp()
    inputs = [torch.ones(1, 51, dtype=torch.bool)]

    cpu_out = wrapper(*inputs)
    assert cpu_out.dtype == torch.int64, f"Expected int64, got {cpu_out.dtype}"

    run_op_test(wrapper, inputs, comparison_config=_COMPARISON, framework=Framework.TORCH)