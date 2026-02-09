# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch_xla.runtime as xr
from tt_torch import codegen_py

# Set up XLA runtime for TT backend.
xr.set_device_type("TT")

class sanity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, anchors):
        output = anchors < 0.99
        return output

model = sanity().to(torch.bfloat16)
model.eval()
anchors = torch.load("anchors.pt",map_location="cpu")

codegen_py(model, anchors, export_path="less_than")
