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
from third_party.tt_forge_models.mlp_mixer.pytorch.loader import ModelLoader,ModelVariant
from loguru import logger

# Set up XLA runtime for TT backend.
xr.set_device_type("TT")


# Any compile options you could specify when executing the model normally can also be used with codegen.
extra_options = {
    # "enable_optimizer": True,
    # "enable_memory_layout_analysis": True,
    # "enable_l1_interleaved": False,
}

loader = ModelLoader(variant=ModelVariant.MIXER_B16_224_GOOG_IN21K)
model = loader.load_model(dtype_override=torch.bfloat16)
x = loader.load_inputs()

logger.info("x={}",x)

logger.info("x1 shape={}",x["x1"].shape)
logger.info("x2 shape={}",x["x2"].shape)

logger.info("x1 dtype={}",x["x1"].dtype)
logger.info("x2 dtype={}",x["x2"].dtype)

codegen_py(model, **x, export_path="mlp_mixer_sanity_block", compiler_options=extra_options)
