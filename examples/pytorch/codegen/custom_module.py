# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Demonstrates how to hook into compile options to use Codegen, from Torch
"""

import os

import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

# Set up XLA runtime for TT backend.
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


# Set up compile options to trigger code generation.
options = {
    # Code generation options
    "backend": "codegen_py",
    # Optimizer options
    # "enable_optimizer": True,
    # "enable_memory_layout_analysis": True,
    # "enable_l1_interleaved": False,
    # Tensor dumping options
    # "dump_inputs": True,
    "export_path": "model",
}
torch_xla.set_custom_compile_options(options)

# Compile for TT, then move the model and it's inputs to device.
device = xm.xla_device()
model = Model()
model.compile(backend="tt")
model = model.to(device)
x = torch.randn(32, 32).to(device)

# Run the model. This triggers code generation.
output = model(x)
