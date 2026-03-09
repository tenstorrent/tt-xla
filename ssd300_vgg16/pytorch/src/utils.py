# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from typing import Optional

import torch
from torch import Tensor


# Patched versions of DefaultBoxGenerator methods that propagate device into
# _grid_default_boxes, so tensors created during forward are on the same device
# (XLA) as the feature maps instead of defaulting to CPU - https://github.com/tenstorrent/tt-xla/issues/3335
# Original forward available at:
# https://github.com/pytorch/vision/blob/v0.24.0/torchvision/models/detection/anchor_utils.py


def patched_grid_default_boxes(
    self,
    grid_sizes: list[list[int]],
    image_size: list[int],
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    default_boxes = []
    for k, f_k in enumerate(grid_sizes):
        if self.steps is not None:
            x_f_k = image_size[1] / self.steps[k]
            y_f_k = image_size[0] / self.steps[k]
        else:
            y_f_k, x_f_k = f_k

        shifts_x = ((torch.arange(0, f_k[1], device=device) + 0.5) / x_f_k).to(
            dtype=dtype
        )
        shifts_y = ((torch.arange(0, f_k[0], device=device) + 0.5) / y_f_k).to(
            dtype=dtype
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)

        shifts = torch.stack(
            (shift_x, shift_y) * len(self._wh_pairs[k]), dim=-1
        ).reshape(-1, 2)
        _wh_pair = self._wh_pairs[k].to(device=device)
        _wh_pair = _wh_pair.clamp(min=0, max=1) if self.clip else _wh_pair
        wh_pairs = _wh_pair.repeat((f_k[0] * f_k[1]), 1)

        default_box = torch.cat((shifts, wh_pairs), dim=1)
        default_boxes.append(default_box)

    return torch.cat(default_boxes, dim=0)


def patched_forward(self, image_list, feature_maps: list[Tensor]) -> list[Tensor]:
    grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
    image_size = image_list.tensors.shape[-2:]
    dtype, device = feature_maps[0].dtype, feature_maps[0].device
    default_boxes = self._grid_default_boxes(
        grid_sizes, image_size, dtype=dtype, device=device
    )
    default_boxes = default_boxes.to(device)

    dboxes = []
    x_y_size = torch.tensor([image_size[1], image_size[0]], device=default_boxes.device)
    for _ in image_list.image_sizes:
        dboxes_in_image = default_boxes
        dboxes_in_image = torch.cat(
            [
                (dboxes_in_image[:, :2] - 0.5 * dboxes_in_image[:, 2:]) * x_y_size,
                (dboxes_in_image[:, :2] + 0.5 * dboxes_in_image[:, 2:]) * x_y_size,
            ],
            -1,
        )
        dboxes.append(dboxes_in_image)
    return dboxes


# See https://github.com/tenstorrent/tt-metal/issues/39171
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

    original_image_sizes: list[tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        torch._assert(
            len(val) == 2,
            f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
        )
        original_image_sizes.append((val[0], val[1]))

    images, targets = self.transform(images, targets)

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

    features = self.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    features = list(features.values())

    head_outputs = self.head(features)

    anchors = self.anchor_generator(images, features)

    return (head_outputs, anchors)
