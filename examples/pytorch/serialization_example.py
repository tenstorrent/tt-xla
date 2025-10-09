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
from tt_torch import parse_compiled_artifacts_from_cache_to_disk

xr.set_device_type("TT")

cache_dir = f"{os.getcwd()}/cachedir"

xr.initialize_cache(cache_dir)


class SimpleModel(nn.Module):
    def forward(self, x, y):
        return x + y


device = xm.xla_device()
model = SimpleModel().to(device)
x = torch.randn(3, 4).to(device)
y = torch.randn(3, 4).to(device)
output = model(x, y)
output.to("cpu")


parse_compiled_artifacts_from_cache_to_disk(cache_dir, "output/model")
