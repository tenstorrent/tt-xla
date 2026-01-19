# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UNet model implementation
Reference: https://github.com/arief25ramadhan/carvana-unet-segmentation
"""

import torch
import torch.nn as nn
from torchvision.transforms import functional


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=1, list_feature_size=[64, 128, 256, 512]
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of unet
        for feature_size in list_feature_size:
            self.downs.append(DoubleConv(in_channels, feature_size))
            in_channels = feature_size

        # Up part of unet
        for feature_size in reversed(list_feature_size):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature_size * 2, feature_size, kernel_size=2, stride=2
                )
            )

            self.ups.append(DoubleConv(feature_size * 2, feature_size))
            in_channels = feature_size

        self.bottleneck = DoubleConv(list_feature_size[-1], list_feature_size[-1] * 2)
        self.final_conv = nn.Conv2d(list_feature_size[0], out_channels, kernel_size=1)

    def forward(self, x):
        list_skip_connections = []

        for down in self.downs:
            x = down(x)
            list_skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        list_skip_connections = list_skip_connections[::-1]

        concat_skip = torch.empty((1, 1))

        for idx, up in enumerate(self.ups):
            if idx % 2 == 0:
                x = up(x)
                skip_connection = list_skip_connections[idx // 2]
                if x.shape != skip_connection.shape:
                    x = functional.resize(x, size=skip_connection.shape[2:])
                concat_skip = torch.cat((skip_connection, x), dim=1)
            else:
                x = up(concat_skip)

        return self.final_conv(x)
