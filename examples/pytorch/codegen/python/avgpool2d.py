# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch_xla.runtime as xr
from tt_torch import codegen_py
from loguru import logger

# Set up XLA runtime for TT backend.
xr.set_device_type("TT")

class avgpool2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False)

    def forward(self, x ):
        return self.pool(x)


model = avgpool2d().to(torch.bfloat16)
model.eval()

inputs = torch.load('avgpool2d_ip_inception.pt',map_location="cpu")
pool = model.pool

logger.info("inputs={}",inputs)
logger.info("inputs.shape={}",inputs.shape)
logger.info("inputs.dtype={}",inputs.dtype)
logger.info("model.pool={}", pool)
logger.info(
    "avgpool2d params: kernel_size={}, stride={}, padding={}, ceil_mode={}, "
    "count_include_pad={}, divisor_override={}",
    pool.kernel_size,
    pool.stride,
    pool.padding,
    pool.ceil_mode,
    pool.count_include_pad,
    pool.divisor_override,
)

codegen_py(model, inputs, export_path="avgpool2d_inception")
