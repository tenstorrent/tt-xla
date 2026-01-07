# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Demonstrates codegen for ResNet-50 from HuggingFace

import torch
from loguru import logger
from tt_torch import codegen_py

import math
import torch.nn.functional as F

def get_same_padding(x: int, kernel_size: int, stride: int, dilation: int):
    return max((math.ceil(x / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - x, 0)

class MaxPool2D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_size = (3, 3)
        self.stride = (2, 2)
        self.dilation=(1, 1)
        self.ceil_mode = False
        
    def forward(self, x):
        x = F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
        ih, iw = x.size()[-2:]
        pad_h = get_same_padding(ih, self.kernel_size[0], self.stride[0], self.dilation[0])
        pad_w = get_same_padding(iw, self.kernel_size[1], self.stride[1], self.dilation[1])
        x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=-float('inf'))
        x = F.max_pool2d(x, self.kernel_size, self.stride, (0, 0), self.dilation, self.ceil_mode)
        return x
        
model = MaxPool2D().to(torch.bfloat16)

logger.info("model={}",model)

x = torch.randn(1, 88, 21, 21,dtype=torch.bfloat16)

codegen_py(model,x , export_path="maxpool2d_pad_maxpool2d_emitpy",export_tensors=False)