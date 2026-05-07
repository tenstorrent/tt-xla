# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch_xla.runtime as xr
from tt_torch import codegen_py
from loguru import logger

# Set up XLA runtime for TT backend.
xr.set_device_type("TT")

class lt(torch.nn.Module):
    def forward(self, anchors ):
        return anchors < 0.99

model = lt()

ip = torch.load("anchor.pt",map_location="cpu")

logger.info("ip={}",ip)
logger.info("ip.shape={}",ip.shape)
logger.info("ip.dtype={}",ip.dtype)

codegen_py(model, ip, export_path="less_than")

