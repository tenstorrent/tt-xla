# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Demonstrates how to hook into serialization to use Codegen(internally also known as EmitC/EmitPy), from Torch
### You should strongly prefer using codegen via compile options
### But for completeness we show how to do it via serialization too

import os

import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

# Set up XLA runtime for TT backend
xr.set_device_type("TT")


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = nn.Linear(32, 128)
        self.B = nn.Linear(128, 64)

    def forward(self, x):
        x = self.A(x)
        x = nn.functional.relu(x)
        x = self.B(x)
        x = torch.tanh(x)
        return torch.sum(x**2)


options = {
    "backend": "codegen_py",
    "export_path": "torch_codegen_example",
}
torch_xla.set_custom_compile_options(options)

device = xm.xla_device()
model = Model()
model.compile(backend="tt")
model = model.to(device)
x = torch.randn(32, 32).to(device)

output = model(x)
