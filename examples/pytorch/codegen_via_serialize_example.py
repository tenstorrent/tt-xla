# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Demonstrates how to hook into serialization to use Codegen(internally also known as EmitC/EmitPy), from Torch
### You should strongly prefer using codegen via compile options
### But for completeness we show how to do it via serialization too

import os

import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import tt_alchemist
from tt_torch import parse_compiled_artifacts_from_cache_to_disk

# Set up XLA runtime for TT backend
xr.set_device_type("TT")

cache_dir = f"{os.getcwd()}/cache_pytorch_codegen"
xr.initialize_cache(cache_dir)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = nn.Linear(32, 128)
        self.B = nn.Linear(128, 64)

    def forward(self, x):
        x = self.A(x)
        x = nn.functional.silu(x)
        x = self.B(x)
        x = torch.tanh(x)
        return torch.sum(x**2)


# Create model and input on TT device
device = xm.xla_device()
model = Model().to(device)
x = torch.randn(32, 32).to(device)

# Execute the model to trigger compilation and caching
output = model(x)
output.to("cpu")

# Parse compiled artifacts from cache to disk
parse_compiled_artifacts_from_cache_to_disk(cache_dir, "model/model")

# Generate C++ code from the TTIR
tt_alchemist.generate_cpp(
    input_file="model/model_ttir.mlir",
    output_dir="model/cpp",
    local=False,
)

# Generate Python code from the TTIR
tt_alchemist.generate_python(
    input_file="model/model_ttir.mlir",
    output_dir="model/py",
    local=False,
)
