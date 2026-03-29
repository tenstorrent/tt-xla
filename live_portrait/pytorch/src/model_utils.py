# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal LivePortrait model architecture definitions for the AppearanceFeatureExtractor.
Adapted from: https://github.com/KwaiVGI/LivePortrait
"""

import torch
from torch import nn
import torch.nn.functional as F


class SameBlock2d(nn.Module):
    """Simple block, preserve spatial resolution."""

    def __init__(
        self, in_features, out_features, groups=1, kernel_size=3, padding=1, lrelu=False
    ):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
        )
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        if lrelu:
            self.ac = nn.LeakyReLU()
        else:
            self.ac = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.ac(out)
        return out


class DownBlock2d(nn.Module):
    """Downsampling block for use in encoder."""

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
        )
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class ResBlock3d(nn.Module):
    """Res block, preserve spatial resolution."""

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels=in_features,
            out_channels=in_features,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2 = nn.Conv3d(
            in_channels=in_features,
            out_channels=in_features,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm1 = nn.BatchNorm3d(in_features, affine=True)
        self.norm2 = nn.BatchNorm3d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class AppearanceFeatureExtractor(nn.Module):
    """Appearance extractor (F) from LivePortrait paper.

    Maps a source image to a 3D appearance feature volume.
    """

    def __init__(
        self,
        image_channel,
        block_expansion,
        num_down_blocks,
        max_features,
        reshape_channel,
        reshape_depth,
        num_resblocks,
    ):
        super(AppearanceFeatureExtractor, self).__init__()
        self.image_channel = image_channel
        self.block_expansion = block_expansion
        self.num_down_blocks = num_down_blocks
        self.max_features = max_features
        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth

        self.first = SameBlock2d(
            image_channel, block_expansion, kernel_size=(3, 3), padding=(1, 1)
        )

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2**i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(
                DownBlock2d(
                    in_features, out_features, kernel_size=(3, 3), padding=(1, 1)
                )
            )
        self.down_blocks = nn.ModuleList(down_blocks)

        self.second = nn.Conv2d(
            in_channels=out_features,
            out_channels=max_features,
            kernel_size=1,
            stride=1,
        )

        self.resblocks_3d = torch.nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_3d.add_module(
                "3dr" + str(i),
                ResBlock3d(reshape_channel, kernel_size=3, padding=1),
            )

    def forward(self, source_image):
        out = self.first(source_image)  # Bx3x256x256 -> Bx64x256x256

        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        out = self.second(out)
        bs, c, h, w = out.shape  # ->Bx512x64x64

        f_s = out.view(
            bs, self.reshape_channel, self.reshape_depth, h, w
        )  # ->Bx32x16x64x64
        f_s = self.resblocks_3d(f_s)  # ->Bx32x16x64x64
        return f_s


# Default model parameters from models.yaml
APPEARANCE_FEATURE_EXTRACTOR_PARAMS = {
    "image_channel": 3,
    "block_expansion": 64,
    "num_down_blocks": 2,
    "max_features": 512,
    "reshape_channel": 32,
    "reshape_depth": 16,
    "num_resblocks": 6,
}


def load_appearance_feature_extractor(checkpoint_path, device="cpu"):
    """Load the AppearanceFeatureExtractor with pretrained weights."""
    model = AppearanceFeatureExtractor(**APPEARANCE_FEATURE_EXTRACTOR_PARAMS).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model
