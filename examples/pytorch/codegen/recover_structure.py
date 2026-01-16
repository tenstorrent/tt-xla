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


class FirstModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(32, 64, dtype=torch.bfloat16))

    def forward(self, x):
        return torch.matmul(x, self.w)


class SecondModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(64, 128, dtype=torch.bfloat16))

    def forward(self, x):
        return torch.matmul(x, self.w)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = FirstModule()
        self.m2 = SecondModule()

    def forward(self, x):
        x = self.m1(x)
        x = self.m2(x)
        # return torch.sum(x**2)  # problem with simple/ast trace when using pow
        # return torch.sum(x)  # add another op outside of first/second modules
        # ^ torch.sum seems to be ignored, doesn't appear in the IR
        return x * x  # add another op outside of first/second modules


# Any compile options you could specify when executing the model normally can also be used with codegen.
extra_options = {
    "codegen_try_recover_structure": True,  # experimental feature
    "export_tensors": True,
}

model = Model()
x = torch.randn(32, 32, dtype=torch.bfloat16)

codegen_py(
    model, x, export_path="recover_structure_example", compiler_options=extra_options
)
