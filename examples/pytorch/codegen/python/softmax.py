# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch_xla.runtime as xr
from tt_torch import codegen_py

# Set up XLA runtime for TT backend.
xr.set_device_type("TT")

class softmax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x ):
        return nn.functional.softmax(x, dim=-1)


model = softmax()
model.eval()

x = torch.randn(1, 100, 6800, dtype=torch.bfloat16)

codegen_py(model, x, export_path="softmax", export_tensors=False)

