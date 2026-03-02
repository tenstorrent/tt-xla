# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Sanity test to isolate the RPN filter_proposals hang in Faster R-CNN.

Reproduces the exact input shapes observed during TT run:
  proposals:            torch.Size([1, 242991, 4])
  objectness:           torch.Size([242991, 1])
  image_shapes:         [(800, 1200)]
  num_anchors_per_level: [182400, 45600, 11400, 2850, 741]
"""

import pytest
import torch
import torchvision


@pytest.fixture
def rpn():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    model.eval()
    return model.rpn

# proposals shape is  torch.Size([1, 242991, 4]) torch.Size([242991, 1]) [(800, 1200)] [182400, 45600, 11400, 2850, 741]
def test_filter_proposals_cpu(rpn):
    device = torch.device("cpu")
    rpn = rpn.to(device)

    proposals = torch.rand(1, 242991, 4, device=device) * 800.0
    objectness = torch.randn(242991, 1, device=device)
    image_shapes = [(800, 1200)]
    num_anchors_per_level = [182400, 45600, 11400, 2850, 741]

    boxes, scores = rpn.filter_proposals(proposals, objectness, image_shapes, num_anchors_per_level)

    assert len(boxes) == 1
    assert len(scores) == 1
    assert boxes[0].shape[-1] == 4


def test_filter_proposals_tt(rpn):
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    rpn = rpn.to(device)

    proposals = torch.rand(1, 242991, 4, device=device) * 800.0
    objectness = torch.randn(242991, 1, device=device)
    image_shapes = [(800, 1200)]
    num_anchors_per_level = [182400, 45600, 11400, 2850, 741]

    boxes, scores = rpn.filter_proposals(proposals, objectness, image_shapes, num_anchors_per_level)

    assert len(boxes) == 1
    assert len(scores) == 1
    assert boxes[0].shape[-1] == 4
