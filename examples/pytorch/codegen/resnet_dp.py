# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Demonstrates codegen for ResNet-50 from HuggingFace with data parallel

import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from transformers import ResNetForImageClassification
from examples.pytorch.multichip.n300.utils import data_parallel_inference_generic
from infra.utilities.torch_multichip_utils import enable_spmd

# Set up TT backend
xr.set_device_type("TT")
enable_spmd()

# Load ResNet-50 from HuggingFace
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
model.eval()

# Set up codegen options
options = {
    "backend": "codegen_py",
    "export_path": "resnet50_codegen_dp",
}
torch_xla.set_custom_compile_options(options)

# Create random input data for ResNet (batch_size, channels, height, width, dtype)
batch_size = 8
input_data = torch.randn(batch_size, 3, 224, 224, dtype=torch.bfloat16)

# Compile the model for TT backend 
model.compile(backend="tt")

# Run multichip data parallel inference. This triggers code generation.
tt_output = data_parallel_inference_generic(
    model=model, 
    inputs=input_data, 
    batch_dim=0
)