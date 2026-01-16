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
from tt_torch import codegen_py

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


# Any compile options you could specify when executing the model normally can also be used with codegen.
extra_options = {
    # "optimization_level": 0,  # Levels 0, 1, and 2 are supported
}

model = Model()
x = torch.randn(32, 32)

codegen_py(model, x, export_path="model", compiler_options=extra_options)
