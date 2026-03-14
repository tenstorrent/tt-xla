# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Isolated sanity tests for the three mask-computation ops that run between
embed_prefix and paligemma_with_expert.forward in sample_actions.

Each test wraps a single op and runs it through run_op_test with the saved
inputs from a prior CPU debug run.  This pinpoints which op (if any)
introduces a PCC drop on TT device.

1. test_make_att_2d_masks
   make_att_2d_masks(prefix_pad_masks, prefix_att_masks)

2. test_cumsum_position_ids
   torch.cumsum(prefix_pad_masks, dim=1) - 1

3. test_prepare_attention_masks_4d
   PI0Pytorch._prepare_attention_masks_4d(prefix_att_2d_masks)

Run the full model with PI0_DEBUG_SAVE_DIR set to generate the .pt files
before running these tests.
"""

import os

import pytest
import torch
from infra import Framework, run_op_test
from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks
from lerobot.utils.constants import OPENPI_ATTENTION_MASK_VALUE

_DEBUG_DIR = os.environ.get("PI0_DEBUG_SAVE_DIR", "debug_folder/pi0_debug")


def _load_saved_data():
    path = os.path.join(_DEBUG_DIR, "block2_paligemma_forward_inputs.pt")
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Op 1: make_att_2d_masks
# ---------------------------------------------------------------------------
class MakeAtt2dMasksOp(torch.nn.Module):
    def forward(self, pad_masks, att_masks):
        return make_att_2d_masks(pad_masks, att_masks)


@pytest.mark.single_device
def test_make_att_2d_masks():
    data = _load_saved_data()
    wrapper = MakeAtt2dMasksOp()
    inputs = [data["prefix_pad_masks"], data["prefix_att_masks"]]
    run_op_test(wrapper, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Op 2: torch.cumsum(pad_masks, dim=1) - 1
# ---------------------------------------------------------------------------
class CumsumPositionIdsOp(torch.nn.Module):
    def forward(self, pad_masks):
        return torch.cumsum(pad_masks, dim=1) - 1


@pytest.mark.single_device
def test_cumsum_position_ids():
    data = _load_saved_data()
    wrapper = CumsumPositionIdsOp()
    inputs = [data["prefix_pad_masks"]]
    run_op_test(wrapper, inputs, framework=Framework.TORCH)


# ---------------------------------------------------------------------------
# Op 3: _prepare_attention_masks_4d
# ---------------------------------------------------------------------------
class PrepareAttentionMasks4dOp(torch.nn.Module):
    def forward(self, att_2d_masks):
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)


@pytest.mark.single_device
def test_prepare_attention_masks_4d():
    data = _load_saved_data()
    prefix_att_2d_masks = make_att_2d_masks(
        data["prefix_pad_masks"], data["prefix_att_masks"]
    )
    wrapper = PrepareAttentionMasks4dOp()
    inputs = [prefix_att_2d_masks]
    run_op_test(wrapper, inputs, framework=Framework.TORCH)
