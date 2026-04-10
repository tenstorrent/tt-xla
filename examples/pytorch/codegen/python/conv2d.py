# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch_xla.runtime as xr
from tt_torch import codegen_py

# Set up XLA runtime for TT backend.
xr.set_device_type("TT")

class conv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(
            128, 128, kernel_size=3, stride=2, padding=1, bias=False
        )

    def forward(self, x):
        return self.conv2d(x)

model = conv()
model.eval()
x = torch.randn(6, 128, 120, 200, dtype=torch.float32)

codegen_py(model, x, export_path="conv2d",export_tensors=False)

