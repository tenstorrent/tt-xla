# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from .aggregator import Aggregator
from .heads.camera_head import CameraHead
from .heads.dpt_head import DPTHead
from .utils.pose_enc import pose_encoding_to_extri_intri

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


def _preprocess_image(images):
    """Normalize images with ImageNet mean/std."""
    mean = torch.tensor(_RESNET_MEAN, device=images.device, dtype=images.dtype).view(
        1, 3, 1, 1
    )
    std = torch.tensor(_RESNET_STD, device=images.device, dtype=images.dtype).view(
        1, 3, 1, 1
    )
    return (images - mean) / std


def _depth_to_points(depth, intrinsics):
    """Convert depth map to 3D points using camera intrinsics."""
    B, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype

    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]

    u = torch.arange(W, device=device, dtype=dtype).view(1, 1, W).expand(B, H, W)
    v = torch.arange(H, device=device, dtype=dtype).view(1, H, 1).expand(B, H, W)

    x = (u - cx.view(B, 1, 1)) * depth / fx.view(B, 1, 1)
    y = (v - cy.view(B, 1, 1)) * depth / fy.view(B, 1, 1)

    return torch.stack([x, y, depth], dim=-1)


class VGGT4Track(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()

        self.aggregator = Aggregator(
            img_size=img_size, patch_size=patch_size, embed_dim=embed_dim
        )
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.depth_head = DPTHead(
            dim_in=2 * embed_dim,
            output_dim=2,
            activation="exp",
            conf_activation="sigmoid",
        )

    def forward(self, images: torch.Tensor, **kwargs):
        """
        Forward pass of the VGGT4Track model.

        Args:
            images: Input images [B, S, 3, H, W] in range [0, 1].

        Returns:
            dict with pose_enc, depth, depth_conf, points_map, etc.
        """
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        B, T, C, H, W = images.shape
        images_proc = _preprocess_image(images.reshape(B * T, C, H, W))
        images_proc = images_proc.reshape(B, T, C, H, W)
        _, _, _, H_proc, W_proc = images_proc.shape

        aggregated_tokens_list, patch_start_idx = self.aggregator(images_proc)

        predictions = {}

        with torch.amp.autocast("cuda", enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list,
                    images=images_proc,
                    patch_start_idx=patch_start_idx,
                )
                predictions["depth"] = depth
                predictions["unc_metric"] = depth_conf.view(B * T, H_proc, W_proc)

        predictions["images"] = images * 255.0

        # Compute camera poses
        poses = torch.eye(4, device=images.device)[None].repeat(T, 1, 1)[None]
        poses[:, :, :3, :4], intrs = pose_encoding_to_extri_intri(
            predictions["pose_enc_list"][-1], images_proc.shape[-2:]
        )
        predictions["poses_pred"] = torch.inverse(poses)
        predictions["intrs"] = intrs

        # Compute 3D point maps
        points_map = _depth_to_points(
            depth.view(B * T, H_proc, W_proc), intrs.view(B * T, 3, 3)
        )
        predictions["points_map"] = F.interpolate(
            points_map.permute(0, 3, 1, 2),
            size=(H, W),
            mode="bilinear",
            align_corners=True,
        ).permute(0, 2, 3, 1)

        predictions["unc_metric"] = F.interpolate(
            predictions["unc_metric"][:, None],
            size=(H, W),
            mode="bilinear",
            align_corners=True,
        )[:, 0]

        predictions["intrs"][..., :1, :] *= W / W_proc
        predictions["intrs"][..., 1:2, :] *= H / H_proc

        return predictions
