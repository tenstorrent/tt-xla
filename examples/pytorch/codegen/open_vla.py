# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Demonstrates codegen for ResNet-50 from HuggingFace

import torch_xla.runtime as xr
from tt_torch import codegen_py
import torch

from third_party.tt_forge_models.openvla.pytorch import ModelVariant,ModelLoader
# Set up XLA runtime for TT backend
xr.set_device_type("TT")

loader = ModelLoader(ModelVariant.OPENVLA_7B)
model = loader.load_model(dtype_override=torch.bfloat16)
model.eval()
x = loader.load_inputs(dtype_override=torch.bfloat16)

codegen_py(model, **x, export_path="openvla_emitpy", export_tensors=False)
