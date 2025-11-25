# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Demonstrates how to hook into compile options to use Codegen, from Torch
"""

import os

import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from transformers import (
    YolosForObjectDetection,
)
from loguru import logger

# Set up XLA runtime for TT backend.
xr.set_device_type("TT")


# Set up compile options to trigger code generation.
options = {
    # Code generation options
    "backend": "codegen_py",
    # Optimizer options
    # "enable_optimizer": True,
    # "enable_memory_layout_analysis": True,
    # "enable_l1_interleaved": False,
    # Tensor dumping options
    # "export_tensors": True,
    "export_path": "linear_sigmoid",
}
torch_xla.set_custom_compile_options(options)

# Compile for TT, then move the model and it's inputs to device.
device = xm.xla_device()


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.linear_2 = model.bbox_predictor.layers[2]
        

    def forward(self, r2):
        
        o3 = self.linear_2(r2)
        pred_boxes = o3.sigmoid()
        
        return pred_boxes

model_kwargs = {"return_dict": False}
model_kwargs["torch_dtype"] = torch.bfloat16

model = YolosForObjectDetection.from_pretrained(
    "hustvl/yolos-small", **model_kwargs
)
model.eval()

model = Wrapper(model)

logger.info("model={}",model)


model.compile(backend="tt")
model = model.to(device)


l2_ip = torch.load('l2_ip_xla.pt',map_location="cpu").to(device)

logger.info("l2_ip={}",l2_ip)
logger.info("l2_ip.dtype={}",l2_ip.dtype)
logger.info("l2_ip.shape={}",l2_ip.shape)

# Run the model. This triggers code generation.
output = model(l2_ip)
