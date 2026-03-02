# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Sanity test for batched_nms from the Faster R-CNN RPN filter_proposals.

Shapes taken from CPU inference logs:
  boxes:  torch.Size([4741, 4])
  scores: torch.Size([4741])
  lvl:    torch.Size([4741])
  nms_thresh: 0.7
"""

import pytest
import torch
from torchvision.ops import boxes as box_ops


def get_tt_device():
    import torch_xla.core.xla_model as xm
    return xm.xla_device()


@pytest.fixture(params=["cpu", "tt"])
def device(request):
    if request.param == "cpu":
        return torch.device("cpu")
    return get_tt_device()


def _make_inputs(device):
    xy = torch.rand(4741, 2, device=device) * 800.0
    wh = torch.rand(4741, 2, device=device) * 50.0 + 1.0
    boxes = torch.cat([xy, xy + wh], dim=1)
    scores = torch.rand(4741, device=device)
    lvl = torch.randint(0, 5, (4741,), device=device)
    return boxes, scores, lvl


def test_batched_nms(device):
    boxes, scores, lvl = _make_inputs(device)
    keep = box_ops.batched_nms(boxes, scores, lvl, 0.7)
    assert keep.ndim == 1
    assert keep.shape[0] <= 4741
