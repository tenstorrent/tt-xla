# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Standalone reproduction of the RPN concat L1 overflow.

Replicates the exact RPN.forward() -> filter_proposals sequence using
torchvision utilities directly. No modifications to installed packages.

The L1 overflow on ConcatDeviceOperation only manifests when the full
graph is compiled together -- every individual op passes in isolation.
"""

import pytest
import torch
import torchvision
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.rpn import concat_box_prediction_layers


FEATURE_SHAPES = {
    "0": (1, 256, 200, 304),
    "1": (1, 256, 100, 152),
    "2": (1, 256, 50, 76),
    "3": (1, 256, 25, 38),
    "pool": (1, 256, 13, 19),
}
IMAGE_TENSOR_SHAPE = (1, 3, 800, 1216)
IMAGE_SIZES = [(800, 1200)]


def _make_inputs(device):
    images = ImageList(
        torch.rand(IMAGE_TENSOR_SHAPE, device=device),
        IMAGE_SIZES,
    )
    features = {k: torch.randn(s, device=device) for k, s in FEATURE_SHAPES.items()}
    return images, features


@pytest.fixture
def rpn():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    model.eval()
    return model.rpn


def rpn_forward_until_loop(rpn, images, features, device):
    """Exact copy of RPN.forward() + filter_proposals up to for-loop iteration."""

    # ---- RPN.forward() ----

    features = list(features.values())
    objectness, pred_bbox_deltas = rpn.head(features)
    anchors = rpn.anchor_generator(images, features)

    num_images = len(anchors)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)

    proposals = rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)

    # ---- filter_proposals() ----

    num_images = proposals.shape[0]
    objectness = objectness.detach()
    objectness = objectness.reshape(num_images, -1)

    levels = [
        torch.full((n,), idx, dtype=torch.int64, device=device)
        for idx, n in enumerate(num_anchors_per_level)
    ]
    levels = torch.cat(levels, 0)
    levels = levels.reshape(1, -1).expand_as(objectness)

    top_n_idx = rpn._get_top_n_idx(objectness, num_anchors_per_level)

    image_range = torch.arange(num_images, device=device)
    batch_idx = image_range[:, None]

    objectness = objectness[batch_idx, top_n_idx]
    levels = levels[batch_idx, top_n_idx]
    proposals = proposals[batch_idx, top_n_idx]

    objectness_prob = torch.sigmoid(objectness)

    # ---- for-loop iteration: proposals[0] triggers L1 overflow ----

    for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, images.image_sizes):
        return boxes


def test_rpn_graph_cpu(rpn):
    device = torch.device("cpu")
    rpn = rpn.to(device)
    images, features = _make_inputs(device)
    boxes = rpn_forward_until_loop(rpn, images, features, device)
    assert boxes.shape == torch.Size([4741, 4])


def test_rpn_graph_tt(rpn):
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    rpn = rpn.to(device)
    images, features = _make_inputs(device)
    boxes = rpn_forward_until_loop(rpn, images, features, device)
    boxes_cpu = boxes.cpu()
    assert boxes_cpu.shape == torch.Size([4741, 4])
