# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Model reference implementation for DenseUNet3d : https://github.com/NguyenJus/pytorch-dense-unet-3d
"""

import torch
from torch import nn
from typing import Tuple


class UpsamplingBlock(nn.Module):
    """
    Upsampling Block (upsampling layer) as specified by the paper
    This is composed of a 2d bilinear upsampling layer followed by a convolutional layer, BatchNorm layer, and ReLU activation
    """

    def __init__(self, in_channels: int, out_channels: int, size: Tuple):
        """
        Create the layers for the upsampling block

        :param in_channels:   number of input features to the block
        :param out_channels:  number of output features from this entire block
        :param scale_factor:  tuple to determine how to scale the dimensions
        :param residual:      residual from the opposite dense block to add before upsampling
        """
        super().__init__()
        # blinear vs trilinear kernel size and padding
        if size[0] == 2:
            d_kernel_size = 3
            d_padding = 1
        else:
            d_kernel_size = 1
            d_padding = 0

        self.upsample = nn.Upsample(
            scale_factor=size, mode="trilinear", align_corners=True
        )
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(d_kernel_size, 3, 3),
                padding=(d_padding, 1, 1),
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, projected_residual):
        """
        Forward pass through the block

        :param x:  image tensor
        :return:   output of the forward pass
        """
        residual = torch.cat(
            (self.upsample(x), self.upsample(projected_residual)),
            dim=1,
        )
        return self.conv(residual)


class TransitionBlock(nn.Module):
    """
    Transition Block (transition layer) as specified by the paper
    This is composed of a pointwise convolution followed by a pointwise convolution with higher stride to reduce the image size
    We use BatchNorm and ReLU after the first convolution, but not the second

    Some notes on architecture based on the paper:
      - The number of channels is always 32
      - The depth is always 3
    """

    def __init__(self, channels: int):
        """
        Create the layers for the transition block

        :param channels:  number of input and output channels, which should be equal
        """
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(channels),
            nn.ReLU(),
            # This conv layer is analogous to H-Dense-UNet's 1x2x2 average pool
            nn.Conv3d(channels, channels, kernel_size=(1, 1, 1), stride=(1, 2, 2)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the block

        :param x:  image tensor
        :return:   output of the forward pass
        """
        return self.convs(x)


class DenseBlock(nn.Module):
    """
    Repeatable Dense block as specified by the paper
    This is composed of a pointwise convolution followed by a depthwise separable convolution
    After each convolution is a BatchNorm followed by a ReLU

    Some notes on architecture based on the paper:
      - The first block uses an input channel of 96, and the remaining input channels are 32
      - The hidden channels is always 128
      - The output channels is always 32
      - The depth is always 3
    """

    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int, count: int
    ):
        """
        Create the layers for the dense block

        :param in_channels:      number of input features to the block
        :param hidden_channels:  number of output features from the first convolutional layer
        :param out_channels:     number of output features from this entire block
        :param count:            number of times to repeat
        """
        super().__init__()

        # First iteration takes different number of input channels and does not repeat
        first_block = [
            nn.Conv3d(in_channels, hidden_channels, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(),
            nn.Conv3d(
                hidden_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        ]

        # Remaining repeats are identical blocks
        repeating_block = [
            nn.Conv3d(out_channels, hidden_channels, kernel_size=(1, 1, 1)),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(),
            nn.Conv3d(
                hidden_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        ]

        self.convs = nn.Sequential(
            *first_block,
            *repeating_block * (count - 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the block

        :param x:  image tensor
        :return:   output of the forward pass
        """
        return self.convs(x)


class DenseUNet3d(nn.Module):
    def __init__(self):
        """
        Create the layers for the model
        """
        super().__init__()
        # Initial Layers
        self.conv1 = nn.Conv3d(
            1, 96, kernel_size=(7, 7, 7), stride=2, padding=(3, 3, 3)
        )
        self.bn1 = nn.BatchNorm3d(96)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1))

        # Dense Layers
        self.transition = TransitionBlock(32)
        self.dense1 = DenseBlock(96, 128, 32, 4)
        self.dense2 = DenseBlock(32, 128, 32, 12)
        self.dense3 = DenseBlock(32, 128, 32, 24)
        self.dense4 = DenseBlock(32, 32, 32, 36)

        # Upsampling Layers
        self.upsample1 = UpsamplingBlock(32 + 32, 504, size=(1, 2, 2))
        self.upsample2 = UpsamplingBlock(504 + 32, 224, size=(1, 2, 2))
        self.upsample3 = UpsamplingBlock(224 + 32, 192, size=(1, 2, 2))
        self.upsample4 = UpsamplingBlock(192 + 32, 96, size=(2, 2, 2))
        self.upsample5 = UpsamplingBlock(96 + 96, 64, size=(2, 2, 2))

        # Final output layer
        # Typo in the paper? Says stride = 0 but that's impossible
        self.conv_classifier = nn.Conv3d(64, 3, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model

        :param x:  image tensor
        :return:   output of the forward pass
        """
        residual1 = self.relu(self.bn1(self.conv1(x)))
        residual2 = self.dense1(self.maxpool1(residual1))
        residual3 = self.dense2(self.transition(residual2))
        residual4 = self.dense3(self.transition(residual3))
        output = self.dense4(self.transition(residual4))

        output = self.upsample1(output, output)
        output = self.upsample2(output, residual4)
        output = self.upsample3(output, residual3)
        output = self.upsample4(output, residual2)
        output = self.upsample5(output, residual1)

        output = self.conv_classifier(output)

        return output
