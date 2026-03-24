# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch_xla.runtime as xr
from tt_torch import codegen_py
from loguru import logger
import torch.nn.functional as F

# Set up XLA runtime for TT backend.
xr.set_device_type("TT")


class avgpool2d(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x ):
        return F.avg_pool2d(x, kernel_size=2, stride=2)


model = avgpool2d().to(torch.bfloat16)
model.eval()

inputs = torch.load('avgpool2d_ip.pt',map_location="cpu")

logger.info("inputs={}",inputs)
logger.info("inputs.shape={}",inputs.shape)
logger.info("inputs.dtype={}",inputs.dtype)
logger.info("model={}",model)
    
codegen_py(model, inputs, export_path="avgpool2d_ghostnet")
