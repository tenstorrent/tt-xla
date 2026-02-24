# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch_xla.runtime as xr
from tt_torch import codegen_py
from loguru import logger 

# Set up XLA runtime for TT backend
xr.set_device_type("TT")


class topk(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        _, topk_ind = torch.topk(inputs, 300, dim=1)
        return topk_ind
    
model = topk().to(torch.bfloat16)
model.eval()

inputs = torch.load('topk_ip.pt',map_location="cpu")
    
logger.info("inputs={}",inputs)
logger.info("inputs.shape={}",inputs.shape)
logger.info("inputs.dtype={}",inputs.dtype)

codegen_py(model,inputs,export_path="topk")

