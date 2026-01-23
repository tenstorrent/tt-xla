# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Demonstrates how to hook into compile options to use Codegen, from Torch
"""

import os

import torch
import torch_xla.runtime as xr
from tt_torch import codegen_py

from third_party.tt_forge_models.openpose.v2.pytorch import ModelLoader

# Set up XLA runtime for TT backend.
xr.set_device_type("TT")

loader = ModelLoader()
model = loader.load_model(dtype_override=torch.bfloat16)
model.eval()

inputs = loader.load_inputs(dtype_override=torch.bfloat16)

codegen_py(model, inputs, export_path="conv2d_bn2d_jan23")
