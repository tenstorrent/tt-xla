# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn
import torch_xla.runtime as xr
from tt_torch import codegen_py
import numpy as np

# Set up XLA runtime for TT backend.
xr.set_device_type("TT")

class reshape(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs ):
        index_dims = inputs.shape[1:-1]
        indices = np.prod(index_dims)
        return 


model = reshape()
model.eval()
inputs = torch.randn(1, 224, 224, 256, dtype=torch.bfloat16)

codegen_py(model, inputs, export_path="prod",export_tensors=False)

