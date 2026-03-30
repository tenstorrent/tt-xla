# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://github.com/Shilin-LU/VINE
"""
CustomConvNeXt model definition for VINE watermark decoding.
"""
from torch import nn
from torchvision import models
from huggingface_hub import PyTorchModelHubMixin


class CustomConvNeXt(nn.Module, PyTorchModelHubMixin):
    """ConvNeXt-based decoder for extracting watermarks from images."""

    def __init__(self, secret_size=100, ckpt_path=None, device=None):
        super(CustomConvNeXt, self).__init__()
        self.convnext = models.convnext_base()
        self.convnext.classifier.append(
            nn.Linear(in_features=1000, out_features=secret_size, bias=True)
        )
        self.convnext.classifier.append(nn.Sigmoid())

    def forward(self, x):
        x = self.convnext(x)
        return x
