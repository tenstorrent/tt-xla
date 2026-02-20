# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch_xla.runtime as xr
from tt_torch import codegen_py
from loguru import logger
import numpy as np

# Set up XLA runtime for TT backend
xr.set_device_type("TT")

class s1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        index_dims = inputs.shape[1:-1]
        return np.prod(index_dims)
        

# model
model = s1()
model.to(torch.bfloat16)
model.eval()

# inputs
inputs = torch.randn(1, 224, 224, 256,dtype=torch.bfloat16)

logger.info("inputs={}",inputs)
logger.info("inputs.shape={}",inputs.shape)
logger.info("inputs.dtype={}",inputs.dtype)

codegen_py(model, inputs, export_path="prod")

