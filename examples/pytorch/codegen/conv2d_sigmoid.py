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
from tt_torch import codegen_py

from third_party.tt_forge_models.retinanet.pytorch.loader import ModelLoader,ModelVariant
from loguru import logger

# Set up XLA runtime for TT backend.
xr.set_device_type("TT")

torch.set_grad_enabled(False)

# Any compile options you could specify when executing the model normally can also be used with codegen.
extra_options = {
    # "enable_optimizer": True,
    # "enable_memory_layout_analysis": True,
    # "enable_l1_interleaved": False,
}

loader = ModelLoader(ModelVariant.RETINANET_RN18FPN)
model = loader.load_model(dtype_override=torch.bfloat16)
model.eval()

logger.info("model={}",model)
conv_ip =  torch.load('conv_ip.pt',map_location="cpu")

logger.info("conv_ip ={}",conv_ip )
logger.info("conv_ip.shape={}",conv_ip.shape)
logger.info("conv_ip.dtype={}",conv_ip.dtype)

codegen_py(model, conv_ip, export_path="conv2d_sig_1", compiler_options=extra_options)
