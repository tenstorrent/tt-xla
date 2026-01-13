# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Demonstrates codegen for ResNet-50 from HuggingFace

import torch
import torch_xla.runtime as xr
from tt_torch import codegen_py
from third_party.tt_forge_models.centernet.pytorch import ModelLoader,ModelVariant

# Set up XLA runtime for TT backend
xr.set_device_type("TT")

loader = ModelLoader(ModelVariant.HOURGLASS_COCO)
model = loader.load_model(dtype_override=torch.bfloat16)
inputs = loader.load_inputs(dtype_override=torch.bfloat16)

codegen_py(model, inputs, export_path="conv2d_bn2d")
