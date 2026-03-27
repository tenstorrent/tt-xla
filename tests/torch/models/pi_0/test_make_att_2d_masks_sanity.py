# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Reproduce and isolate the hang observed in the PI0 LIBERO_BASE inference path
when make_att_2d_masks() is called on prefix masks inside sample_actions().

FINDING: make_att_2d_masks in isolation passes (all 8 sub-op tests pass).
The hang occurs only when make_att_2d_masks is **composed** with upstream
embed_prefix computation in a single XLA graph.  The combined StableHLO ->
TTIR compilation hangs.

"""

import pytest
import torch
from infra import ComparisonConfig, Framework, run_op_test
from infra.evaluators import AllcloseConfig
from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks


_COMPARISON = ComparisonConfig(
    allclose=AllcloseConfig(enabled=True, rtol=1e-5, atol=1e-5),
)


# ---------------------------------------------------------------------------
# Op wrappers
# ---------------------------------------------------------------------------
class MakeAtt2dMasksOp(torch.nn.Module):
    """Full make_att_2d_masks as used in modeling_pi0.py (exact copy)."""

    def forward(self, pad_masks, att_masks):
        cumsum = torch.cumsum(att_masks, dim=1)
        att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
        pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
        return att_2d_masks & pad_2d_masks


class BroadcastLeOp(torch.nn.Module):
    """Isolated broadcast <= comparison that expands [B, N] -> [B, N, N].

    cumsum[:, None, :] <= cumsum[:, :, None]
    """

    def forward(self, cumsum):
        return cumsum[:, None, :] <= cumsum[:, :, None]


class BroadcastMulBoolOp(torch.nn.Module):
    """Isolated broadcast multiply on bool masks -> [B, N, N].

    pad_masks[:, None, :] * pad_masks[:, :, None]
    """

    def forward(self, pad_masks):
        return pad_masks[:, None, :] * pad_masks[:, :, None]


class BitwiseAndOp(torch.nn.Module):
    """Isolated bitwise AND on two bool [B, N, N] tensors."""

    def forward(self, a, b):
        return a & b


class CumsumThenBroadcastLeOp(torch.nn.Module):
    """cumsum + broadcast <= composed (steps 1+2)."""

    def forward(self, att_masks):
        cumsum = torch.cumsum(att_masks, dim=1)
        return cumsum[:, None, :] <= cumsum[:, :, None]


# ---------------------------------------------------------------------------
# Full make_att_2d_masks tests  (expected to reproduce hang)
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_make_att_2d_masks_prefix():
    """Full make_att_2d_masks with prefix masks from sample_actions.

    Observed inputs:
      pad_masks: bool[1, 816] – all True  (fully-present inputs)
      att_masks: bool[1, 816] – all False (prefix-LM: no causal masking)
    Output: bool[1, 816, 816]
    """
    wrapper = MakeAtt2dMasksOp()
    pad_masks = torch.ones(1, 816, dtype=torch.bool)
    att_masks = torch.zeros(1, 816, dtype=torch.bool)
    inputs = [pad_masks, att_masks]

    cpu_out = wrapper(*inputs)
    assert cpu_out.shape == (1, 816, 816), f"Expected (1,816,816), got {cpu_out.shape}"
    assert cpu_out.dtype == torch.bool, f"Expected bool, got {cpu_out.dtype}"

    run_op_test(wrapper, inputs, comparison_config=_COMPARISON, framework=Framework.TORCH)


@pytest.mark.single_device
def test_make_att_2d_masks_suffix():
    """Full make_att_2d_masks with suffix masks from denoise_step.

    Observed inputs:
      pad_masks: bool[1, 51]    – all True
      att_masks: float32[1, 51] – [1, 1, 0, 0, ..., 0]
    Output: bool[1, 51, 51]
    """
    wrapper = MakeAtt2dMasksOp()
    pad_masks = torch.ones(1, 51, dtype=torch.bool)
    att_masks = torch.tensor([[1, 1, 0] + [0] * 48], dtype=torch.float32)
    inputs = [pad_masks, att_masks]

    cpu_out = wrapper(*inputs)
    assert cpu_out.shape == (1, 51, 51), f"Expected (1,51,51), got {cpu_out.shape}"

    run_op_test(wrapper, inputs, comparison_config=_COMPARISON, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Sub-op slice tests  (run these after confirming full test hangs)
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_slice_broadcast_le_prefix():
    """Isolated: cumsum[:, None, :] <= cumsum[:, :, None]

    Input:  int64[1, 816] (result of cumsum on all-False bool -> all zeros)
    Output: bool[1, 816, 816]
    """
    wrapper = BroadcastLeOp()
    cumsum = torch.zeros(1, 816, dtype=torch.int64)
    inputs = [cumsum]

    cpu_out = wrapper(*inputs)
    assert cpu_out.shape == (1, 816, 816)
    assert cpu_out.dtype == torch.bool

    run_op_test(wrapper, inputs, comparison_config=_COMPARISON, framework=Framework.TORCH)


@pytest.mark.single_device
def test_slice_broadcast_mul_bool_prefix():
    """Isolated: pad_masks[:, None, :] * pad_masks[:, :, None]

    Input:  bool[1, 816] (all True)
    Output: bool[1, 816, 816]
    """
    wrapper = BroadcastMulBoolOp()
    pad_masks = torch.ones(1, 816, dtype=torch.bool)
    inputs = [pad_masks]

    cpu_out = wrapper(*inputs)
    assert cpu_out.shape == (1, 816, 816)

    run_op_test(wrapper, inputs, comparison_config=_COMPARISON, framework=Framework.TORCH)


@pytest.mark.single_device
def test_slice_bitwise_and_prefix():
    """Isolated: att_2d_masks & pad_2d_masks

    Both inputs: bool[1, 816, 816] (all True in prefix case)
    Output:      bool[1, 816, 816]
    """
    wrapper = BitwiseAndOp()
    a = torch.ones(1, 816, 816, dtype=torch.bool)
    b = torch.ones(1, 816, 816, dtype=torch.bool)
    inputs = [a, b]

    cpu_out = wrapper(*inputs)
    assert cpu_out.shape == (1, 816, 816)
    assert cpu_out.dtype == torch.bool

    run_op_test(wrapper, inputs, comparison_config=_COMPARISON, framework=Framework.TORCH)


@pytest.mark.single_device
def test_slice_cumsum_then_broadcast_le_prefix():
    """Composed: cumsum + broadcast <= (steps 1+2 together).

    Input:  bool[1, 816] (all False)
    Output: bool[1, 816, 816]
    """
    wrapper = CumsumThenBroadcastLeOp()
    att_masks = torch.zeros(1, 816, dtype=torch.bool)
    inputs = [att_masks]

    cpu_out = wrapper(*inputs)
    assert cpu_out.shape == (1, 816, 816)
    assert cpu_out.dtype == torch.bool

    run_op_test(wrapper, inputs, comparison_config=_COMPARISON, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Smaller-shape variants (quick smoke test to rule out shape-dependent hangs)
# ---------------------------------------------------------------------------
@pytest.mark.single_device
def test_make_att_2d_masks_small():
    """make_att_2d_masks with a small [1, 8] shape for fast triage."""
    wrapper = MakeAtt2dMasksOp()
    pad_masks = torch.ones(1, 8, dtype=torch.bool)
    att_masks = torch.zeros(1, 8, dtype=torch.bool)
    inputs = [pad_masks, att_masks]

    cpu_out = wrapper(*inputs)
    assert cpu_out.shape == (1, 8, 8)

    run_op_test(wrapper, inputs, comparison_config=_COMPARISON, framework=Framework.TORCH)


@pytest.mark.single_device
def test_slice_broadcast_le_small():
    """Broadcast <= with small [1, 8] for fast triage."""
    wrapper = BroadcastLeOp()
    cumsum = torch.zeros(1, 8, dtype=torch.int64)
    inputs = [cumsum]

    cpu_out = wrapper(*inputs)
    assert cpu_out.shape == (1, 8, 8)

    run_op_test(wrapper, inputs, comparison_config=_COMPARISON, framework=Framework.TORCH)

