# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional
import torch
from torch import Tensor
from collections import OrderedDict


def patched_SSD_forward(
    self, images: list[Tensor], targets: Optional[list[dict[str, Tensor]]] = None
) -> tuple[dict[str, Tensor], list[dict[str, Tensor]]]:
    if self.training:
        if targets is None:
            torch._assert(False, "targets should not be none when in training mode")
        else:
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                    )
                else:
                    torch._assert(
                        False,
                        f"Expected target boxes to be of type Tensor, got {type(boxes)}.",
                    )

    # get the original image sizes
    original_image_sizes: list[tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        torch._assert(
            len(val) == 2,
            f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
        )
        original_image_sizes.append((val[0], val[1]))

    # transform the input
    images, targets = self.transform(images, targets)

    # Check for degenerate boxes
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: list[float] = boxes[bb_idx].tolist()
                torch._assert(
                    False,
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}.",
                )

    # get the features from the backbone
    features = self.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    features = list(features.values())

    # compute the ssd heads outputs using the features
    head_outputs = self.head(features)

    # create the set of anchors
    anchors = self.anchor_generator(images, features)

    return (head_outputs, anchors)
