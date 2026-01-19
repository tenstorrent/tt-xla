# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# This PyTorch implementation is based on: https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/TensorFlow/vgg19_unet.py

import torch
import torch.nn as nn
import torchvision.models as models


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0
        )
        self.conv_block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


class VGG19UNet(nn.Module):
    def __init__(self, input_shape=(3, 512, 512), out_channels=1):
        super(VGG19UNet, self).__init__()

        # Load pre-trained VGG19 model
        vgg19 = models.vgg19(weights="IMAGENET1K_V1")

        # Encoder (VGG19 feature layers)
        self.encoder1 = vgg19.features[:4]  # block1_conv2
        self.encoder2 = vgg19.features[4:9]  # block2_conv2
        self.encoder3 = vgg19.features[9:18]  # block3_conv4
        self.encoder4 = vgg19.features[18:27]  # block4_conv4
        self.bridge = vgg19.features[27:36]  # block5_conv4

        # Freeze encoder weights
        for param in self.encoder1.parameters():
            param.requires_grad = False
        for param in self.encoder2.parameters():
            param.requires_grad = False
        for param in self.encoder3.parameters():
            param.requires_grad = False
        for param in self.encoder4.parameters():
            param.requires_grad = False
        for param in self.bridge.parameters():
            param.requires_grad = False

        # Decoder
        self.decoder1 = DecoderBlock(512, 512, 512)  # (64x64)
        self.decoder2 = DecoderBlock(512, 256, 256)  # (128x128)
        self.decoder3 = DecoderBlock(256, 128, 128)  # (256x256)
        self.decoder4 = DecoderBlock(128, 64, 64)  # (512x512)

        # Output layer
        self.output = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Bridge
        b = self.bridge(e4)

        # Decoder
        d1 = self.decoder1(b, e4)
        d2 = self.decoder2(d1, e3)
        d3 = self.decoder3(d2, e2)
        d4 = self.decoder4(d3, e1)

        # Output
        outputs = self.sigmoid(self.output(d4))

        return outputs
