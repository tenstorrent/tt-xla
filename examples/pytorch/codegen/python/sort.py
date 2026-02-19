
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch_xla.runtime as xr
from tt_torch import codegen_py
from loguru import logger

# Set up XLA runtime for TT backend
xr.set_device_type("TT")

class sort(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sort_ip):
        return sort_ip.sort(descending=True)[1]

# model
model = sort()
model.eval()

# inputs
sort_ip = torch.load('sort_ip.pt',map_location="cpu")

logger.info("sort_ip={}",sort_ip)
logger.info("sort_ip.shape={}",sort_ip.shape)
logger.info("sort_ip.dtype={}",sort_ip.dtype)


codegen_py(model, sort_ip, export_path="sort")
