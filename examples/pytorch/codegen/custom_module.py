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


class TopKIndices(torch.nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        return torch.topk(x, self.k)[1]

# Any compile options you could specify when executing the model normally can also be used with codegen.
extra_options = {
    # "enable_optimizer": True,
    # "enable_memory_layout_analysis": True,
    # "enable_l1_interleaved": False,
}

model = TopKIndices(k=100)
model.eval()
x = torch.load("topk_ip.pt")

codegen_py(model, x, export_path="topk", compiler_options=extra_options)
