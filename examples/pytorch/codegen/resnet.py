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

# Any compile options you could specify when executing the model normally can also be used with codegen.
extra_options = {
    # Optimization level (0, 1, or 2) that controls multiple optimization passes.
    # See documentation for details on what each level enables:
    # https://docs.tenstorrent.com/tt-xla/performance.html#optimization-levels-breakdown
    # Level 0 (default): All optimizations disabled
    # Level 1: Basic optimizations (optimizer + Conv2d fusion)
    # Level 2: Advanced optimizations (optimizer + memory layout + Conv2d fusion)
    "optimization_level": 2,
    # Experimental feature tries to pack generated code as it was originally written,
    # making it more readable.
    "enable_prettify": False,  # experimental feature
}

activation_tensor = torch.randn(1, 3, 224, 224)

codegen_py(
    model,
    activation_tensor,
    export_path="resnet50_codegen",
    compiler_options=extra_options,
)
