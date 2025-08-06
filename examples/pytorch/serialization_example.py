# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from torch_xla.experimental import plugins

from tt_torch import parse_from_cache_to_disk

xr.set_device_type("TT")

xr.initialize_cache(f"{os.getcwd()}/cachedir")


class SimpleModel(nn.Module):
    def forward(self, x, y):
        return x + y


device = xm.xla_device()
model = SimpleModel().to(device)
x = torch.randn(3, 4).to(device)
y = torch.randn(3, 4).to(device)
output = model(x, y)
output.to("cpu")


parse_from_cache_to_disk("output/model", f"{os.getcwd()}/cachedir")
