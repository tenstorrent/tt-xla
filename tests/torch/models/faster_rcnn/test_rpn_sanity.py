# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Sanity test to isolate the RPN module hang in Faster R-CNN.

Reproduces the exact input shapes observed during CPU inference:
  images.tensors:       torch.Size([1, 3, 800, 1216])
  images.image_sizes:   [(800, 1200)]
  features:             {'0': [1,256,200,304], '1': [1,256,100,152],
                         '2': [1,256,50,76],   '3': [1,256,25,38],
                         'pool': [1,256,13,19]}
"""

import pytest
import torch
import torchvision
from torchvision.models.detection.image_list import ImageList


FEATURE_SHAPES = {
    "0": (1, 256, 200, 304),
    "1": (1, 256, 100, 152),
    "2": (1, 256, 50, 76),
    "3": (1, 256, 25, 38),
    "pool": (1, 256, 13, 19),
}
IMAGE_TENSOR_SHAPE = (1, 3, 800, 1216)
IMAGE_SIZES = [(800, 1200)]


@pytest.fixture
def rpn():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    model.eval()
    return model.rpn


def _make_inputs(device):
    images = ImageList(
        torch.rand(IMAGE_TENSOR_SHAPE, device=device),
        IMAGE_SIZES,
    )
    features = {k: torch.randn(s, device=device) for k, s in FEATURE_SHAPES.items()}
    return images, features


def test_rpn_cpu(rpn):
    device = torch.device("cpu")
    rpn = rpn.to(device)
    images, features = _make_inputs(device)

    boxes, losses = rpn(images, features)

    assert len(boxes) == 1
    assert boxes[0].shape[-1] == 4
    assert isinstance(losses, dict)


def test_rpn_tt(rpn):
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    rpn = rpn.to(device)
    images, features = _make_inputs(device)

    boxes, losses = rpn(images, features)

    assert len(boxes) == 1
    assert boxes[0].shape[-1] == 4
    assert isinstance(losses, dict)
