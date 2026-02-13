# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import torch
import torch_xla.runtime as xr
from tt_torch import codegen_py
from third_party.tt_forge_models.yolov9.pytorch.loader import ModelLoader, ModelVariant

# Set up XLA runtime for TT backend.
xr.set_device_type("TT")

loader = ModelLoader(ModelVariant.S)

model = loader.load_model(dtype_override=torch.bfloat16)
x = loader.load_inputs(dtype_override=torch.bfloat16)
codegen_py(model, x, export_path="silu")
