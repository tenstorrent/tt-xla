# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch_xla.runtime as xr
from tt_torch import codegen_py
from third_party.tt_forge_models.mgp_str_base.pytorch import ModelLoader
from loguru import logger

# Set up XLA runtime for TT backend.
xr.set_device_type("TT")

loader = ModelLoader()
model = loader.load_model(dtype_override=torch.bfloat16)

conv_ip = torch.load('conv_ip.pt',map_location="cpu")
        
logger.info("conv_ip={}",conv_ip)
logger.info("conv_ip.shape={}",conv_ip.shape)
logger.info("conv_ip.dtype={}",conv_ip.dtype)

feat = torch.load('feat2.pt',map_location="cpu")

logger.info("feat={}",feat)
logger.info("feat.shape={}",feat.shape)
logger.info("feat.dtype={}",feat.dtype)

inputs = [conv_ip,feat]
        
codegen_py(model, *inputs, export_path="sanity")

