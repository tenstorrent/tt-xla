# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch_xla.runtime as xr
from tt_torch import codegen_py

# Set up XLA runtime for TT backend.
xr.set_device_type("TT")

class AtrousConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(18, 18),
            dilation=(18, 18),
            bias=False,
            groups=1,
        )

    def forward(self, x):
        return self.conv(x)

model = AtrousConv()
model.eval()

x = torch.randn(6, 512, 16, 44, dtype=torch.float32)

codegen_py(model, x, export_path="conv_bevdepth",export_tensors=False)

