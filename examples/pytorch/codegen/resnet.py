# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Demonstrates codegen for ResNet-50 from HuggingFace

import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from transformers import ResNetForImageClassification
from tt_torch import codegen_py

# Set up XLA runtime for TT backend
xr.set_device_type("TT")

# Load ResNet-50 from HuggingFace
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
model.eval()
x = torch.randn(1, 3, 224, 224)

codegen_py(model, x, export_path="resnet50_codegen")
