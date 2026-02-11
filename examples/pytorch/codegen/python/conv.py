# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch_xla.runtime as xr
from tt_torch import codegen_py
from third_party.tt_forge_models.vocoder_speecht5.pytorch import ModelLoader


# Set up XLA runtime for TT backend
xr.set_device_type("TT")

loader = ModelLoader()
model = loader.load_model(dtype_override=torch.bfloat16)
model.eval()
x = loader.load_inputs(dtype_override=torch.bfloat16)

codegen_py(model, x, export_path="conv")

