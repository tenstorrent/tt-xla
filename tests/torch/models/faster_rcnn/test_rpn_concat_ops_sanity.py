# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Sanity tests for the individual torch.cat ops inside the Faster R-CNN RPN
that produce the large 242991-element tensors.

Shapes taken from CPU inference logs.
"""

import pytest
import torch


def get_tt_device():
    import torch_xla.core.xla_model as xm
    return xm.xla_device()


@pytest.fixture(params=["cpu", "tt"])
def device(request):
    if request.param == "cpu":
        return torch.device("cpu")
    return get_tt_device()


def test_cat_box_cls(device):
    """concat_box_prediction_layers: cat box_cls_flattened along dim=1, then flatten."""
    box_cls_flattened = [
        torch.randn(1, 182400, 1, device=device),
        torch.randn(1, 45600, 1, device=device),
        torch.randn(1, 11400, 1, device=device),
        torch.randn(1, 2850, 1, device=device),
        torch.randn(1, 741, 1, device=device),
    ]
    print("before concat is ",box_cls_flattened[0].shape)
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    print("after concat is ",box_cls.shape)
    assert box_cls.shape == torch.Size([242991, 1])


def test_cat_box_regression(device):
    """concat_box_prediction_layers: cat box_regression_flattened along dim=1, then reshape."""
    box_regression_flattened = [
        torch.randn(1, 182400, 4, device=device),
        torch.randn(1, 45600, 4, device=device),
        torch.randn(1, 11400, 4, device=device),
        torch.randn(1, 2850, 4, device=device),
        torch.randn(1, 741, 4, device=device),
    ]
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, 4)
    assert box_regression.shape == torch.Size([242991, 4])


def test_cat_levels(device):
    """filter_proposals: cat levels along dim=0."""
    levels = [
        torch.full((182400,), 0, dtype=torch.int64, device=device),
        torch.full((45600,), 1, dtype=torch.int64, device=device),
        torch.full((11400,), 2, dtype=torch.int64, device=device),
        torch.full((2850,), 3, dtype=torch.int64, device=device),
        torch.full((741,), 4, dtype=torch.int64, device=device),
    ]
    result = torch.cat(levels, 0)
    assert result.shape == torch.Size([242991])


def test_cat_top_n_idx(device):
    """_get_top_n_idx: cat top_n_idx along dim=1."""
    r = [
        torch.randint(0, 182400, (1, 1000), device=device),
        torch.randint(0, 45600, (1, 1000), device=device),
        torch.randint(0, 11400, (1, 1000), device=device),
        torch.randint(0, 2850, (1, 1000), device=device),
        torch.randint(0, 741, (1, 741), device=device),
    ]
    result = torch.cat(r, dim=1)
    assert result.shape == torch.Size([1, 4741])


def test_stack_decode_single(device):
    """decode_single: torch.stack 4 pred_boxes [242991,1] along dim=2, then flatten."""
    pred_boxes1 = torch.randn(242991, 1, device=device)
    pred_boxes2 = torch.randn(242991, 1, device=device)
    pred_boxes3 = torch.randn(242991, 1, device=device)
    pred_boxes4 = torch.randn(242991, 1, device=device)
    result = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
    assert result.shape == torch.Size([242991, 4])


def test_cat_anchors(device):
    """anchor_generator: cat anchors from 5 feature levels along dim=0."""
    anchors = [
        torch.randn(182400, 4, device=device),
        torch.randn(45600, 4, device=device),
        torch.randn(11400, 4, device=device),
        torch.randn(2850, 4, device=device),
        torch.randn(741, 4, device=device),
    ]
    result = torch.cat(anchors, dim=0)
    assert result.shape == torch.Size([242991, 4])


def test_stack_clip_boxes(device):
    """clip_boxes_to_image: torch.stack boxes_x, boxes_y [4741,2] along dim=2."""
    boxes_x = torch.randn(4741, 2, device=device)
    boxes_y = torch.randn(4741, 2, device=device)
    result = torch.stack((boxes_x, boxes_y), dim=2).reshape(4741, 4)
    assert result.shape == torch.Size([4741, 4])


def test_decode_then_index(device):
    """decode_single stack + [0] indexing -- minimal combo that may trigger L1."""
    pred_boxes = [torch.randn(242991, 1, device=device) for _ in range(4)]
    result = torch.stack(pred_boxes, dim=2).flatten(1)
    result = result.view(1, -1, 4)
    result = result[0]
    assert result.shape == torch.Size([242991, 4])


def test_decode_with_gather(device):
    """decode stack + gather (top_n_idx) + [0] indexing -- closer to real RPN graph."""
    pred_boxes = [torch.randn(242991, 1, device=device) for _ in range(4)]
    proposals = torch.stack(pred_boxes, dim=2).flatten(1)
    proposals = proposals.view(1, -1, 4)
    top_n_idx = torch.randint(0, 242991, (1, 4741), device=device)
    batch_idx = torch.arange(1, device=device)[:, None]
    proposals = proposals[batch_idx, top_n_idx]
    boxes = proposals[0]
    assert boxes.shape == torch.Size([4741, 4])
