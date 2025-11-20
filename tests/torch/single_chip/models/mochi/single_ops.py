# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.nn as nn
import torch_xla
import torch_xla.runtime as xr

os.environ["XLA_HLO_DEBUG"] = "1"

"""
Minimal reproduction of Conv3d operation that fails in Mochi decoder.

This isolates the exact Conv3d that causes:
  error: failed to legalize operation 'ttir.convolution' that was explicitly marked illegal

Reproduce the exact Conv3d from MochiDecoder3D.conv_in

From decoder.__init__():
    self.conv_in = nn.Conv3d(in_channels, block_out_channels[-1], kernel_size=(1, 1, 1))

Where:
    in_channels = 12 (latent channels)
    block_out_channels[-1] = 768
    kernel_size = (1, 1, 1) - point-wise convolution
"""


class MochiConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(MochiConv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return self.conv3d(x)


def test_conv3d_reproduction():
    device = torch_xla.device()

    # Create the Conv3d layer
    conv3d = MochiConv3d(in_channels=12, out_channels=768, kernel_size=(1, 1, 1))
    conv3d = conv3d.to(torch.bfloat16).to(device)

    conv3d = torch.compile(conv3d, backend="tt")

    # From decoder: [B, 12, t, h, w]
    # Example: [1, 12, 3, 32, 32] for 13 frames at 256x256
    batch = 1
    channels = 12
    temporal = 3
    height = 32
    width = 32

    x = torch.randn(batch, channels, temporal, height, width, dtype=torch.bfloat16)
    x = x.to(device)

    print(f"\nRunning forward pass...")
    with torch.no_grad():
        output = conv3d(x)
    print(f"✓ Success! Output shape: {output.shape}")
    print(f"Output: {output}")
    return output


def test_conv2d_reproduction():
    device = torch_xla.device()

    # Create the Conv2d layer
    conv2d = nn.Conv2d(in_channels=3, out_channels=768, kernel_size=(1, 1))
    conv2d = conv2d.to(torch.bfloat16).to(device)

    conv2d = torch.compile(conv2d, backend="tt")
    batch = 1
    channels = 3
    height = 32
    width = 32

    x = torch.randn(batch, channels, height, width, dtype=torch.bfloat16)
    x = x.to(device)

    print(f"\nRunning forward pass...")
    with torch.no_grad():
        output = conv2d(x)
    print(f"✓ Success! Output shape: {output.shape}")
    return output


if __name__ == "__main__":
    xr.set_device_type("TT")
    test_conv3d_reproduction()
    # test_conv2d_reproduction()
