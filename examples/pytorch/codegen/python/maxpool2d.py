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

class maxpool2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x ):
        return F.max_pool2d(
            x,
            (3, 3),
            (2, 2),
            (0, 0),
            (1, 1),
            False,
        )


model = maxpool2d().to(torch.bfloat16)
model.eval()

inputs = torch.load('maxpool2d_new_ip.pt',map_location="cpu")
logger.info("inputs={}",inputs)
logger.info("inputs.shape={}",inputs.shape)
logger.info("inputs.dtype={}",inputs.dtype)
logger.info("model={}",model)
    
codegen_py(model,inputs , export_path="maxpool2d")

