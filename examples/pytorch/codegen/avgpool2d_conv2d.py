# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Demonstrates codegen for ResNet-50 from HuggingFace

import torch
import torch_xla.runtime as xr
from loguru import logger

from tt_torch import codegen_py

# Set up XLA runtime for TT backend
xr.set_device_type("TT")

class avgpool2d_conv2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=360,
            out_channels=480,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            groups=1,
        )

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.conv(x)
    
model = avgpool2d_conv2d().to(torch.bfloat16)
logger.info("model={}",model)

x= torch.randn(1, 360, 28, 40, dtype=torch.bfloat16)

codegen_py(model,x , export_path="avgpool2d_conv2d_emitpy",export_tensors=False)