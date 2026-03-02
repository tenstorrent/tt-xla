# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Demonstrates codegen (emitPy) for the RPN module of Faster R-CNN.

The RPN forward expects (ImageList, dict[str, Tensor]) which codegen_py
cannot handle directly (it filters args to torch.Tensor only). This wrapper
takes flat tensors as positional args and reconstructs the structured inputs
internally so codegen_py can trace through the module.
"""

import torch
import torch.nn as nn
import torch_xla.runtime as xr
from tt_torch import codegen_py
import torchvision
from torchvision.models.detection.image_list import ImageList


FEATURE_KEYS = ["0", "1", "2", "3", "pool"]
FEATURE_SHAPES = {
    "0": (1, 256, 200, 304),
    "1": (1, 256, 100, 152),
    "2": (1, 256, 50, 76),
    "3": (1, 256, 25, 38),
    "pool": (1, 256, 13, 19),
}
IMAGE_TENSOR_SHAPE = (1, 3, 800, 1216)
IMAGE_SIZES = [(800, 1200)]


class RPNWrapper(nn.Module):
    """Wraps the RPN to accept flat tensor args for codegen compatibility."""

    def __init__(self, rpn):
        super().__init__()
        self.rpn = rpn

    def forward(self, image_tensor, feat_0, feat_1, feat_2, feat_3, feat_pool):
        images = ImageList(image_tensor, IMAGE_SIZES)
        features = {
            "0": feat_0,
            "1": feat_1,
            "2": feat_2,
            "3": feat_3,
            "pool": feat_pool,
        }
        boxes, losses = self.rpn(images, features)
        return boxes[0]


def test_codegen():
    xr.set_device_type("TT")

    base_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    base_model.eval()

    model = RPNWrapper(base_model.rpn)
    model.eval()

    image_tensor = torch.rand(IMAGE_TENSOR_SHAPE)
    feat_tensors = [torch.randn(FEATURE_SHAPES[k]) for k in FEATURE_KEYS]

    codegen_py(
        model,
        image_tensor, *feat_tensors,
        export_path="faster_rcnn_rpn_module",
    )


if __name__ == "__main__":
    test_codegen()



