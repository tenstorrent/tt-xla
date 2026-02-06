# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from tt_torch import codegen_py
import torch.nn.functional as F

# Set up XLA runtime for TT backend.
xr.set_device_type("TT")

class PadModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        x1 = F.pad(x1, [-16,-16,-16,-16])
        return x1

x1 = torch.randn(1, 1024, 64, 64,dtype=torch.bfloat16)
model = PadModule()
model.to(torch.bfloat16)

codegen_py(model, x1 , export_path="pad_emitpy")