# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Demonstrates how to hook into compile options to use Codegen, from Torch
"""

import os

import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

# Set up XLA runtime for TT backend.
xr.set_device_type("TT")


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

    def forward(self, x1, x2):
        y1 = torch.sigmoid(x2)
        y2 = self.conv(x1)
        return (y1, y2)


# Set up compile options to trigger code generation.
options = {
    # Code generation options
    "backend": "codegen_py",
    # Optimizer options
    # "enable_optimizer": True,
    # "enable_memory_layout_analysis": True,
    # "enable_l1_interleaved": False,
    # Tensor dumping options
    # "export_tensors": True,
    "export_path": "sig_conv2d_randn",
}
torch_xla.set_custom_compile_options(options)

# Compile for TT, then move the model and it's inputs to device.
device = xm.xla_device()
model = Model()
model.compile(backend="tt")
model = model.to(device)
x1 = torch.randn(1, 64, 64, 64).to(device)
x2 = torch.randn(1, 1, 64, 64).to(device)

# Run the model. This triggers code generation.
output = model(x1, x2)
