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
from third_party.tt_forge_models.autoencoder.pytorch import ModelLoader,ModelVariant
from loguru import logger


# Set up XLA runtime for TT backend.
xr.set_device_type("TT")


# Any compile options you could specify when executing the model normally can also be used with codegen.
extra_options = {
    # "enable_optimizer": True,
    # "enable_memory_layout_analysis": True,
    # "enable_l1_interleaved": False,
    
}


loader = ModelLoader(ModelVariant.CONV)

# Load the model
model = loader.load_model(dtype_override=torch.bfloat16)

logger.info("model={}",model)

act = torch.load('encoder_max_pool2d_ip_2.pt',map_location="cpu")

logger.info("act={}",act)
logger.info("act.dtype={}",act.dtype)
logger.info("act.shape={}",act.shape)

codegen_py(model, act , export_path="maxpool2d_conv2dt_dec11", compiler_options=extra_options)
