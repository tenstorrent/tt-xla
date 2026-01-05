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

class MaxPool2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, input_tensor):
        return self.maxpool(input_tensor)

model = MaxPool2d()

logger.info("model={}",model)

x= torch.randn(6, 64, 240 , 400)

codegen_py(model,x , export_path="maxpool2d_emitpy_maptr",export_tensors=False)