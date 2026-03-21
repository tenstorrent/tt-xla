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

class adaptive_avgpool2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool_lidar = nn.AdaptiveAvgPool2d((8, 8))

    def forward(self, lidar_features  ):
        
        output = self.avgpool_lidar(lidar_features)
        return output

model = adaptive_avgpool2d().to(torch.bfloat16)

lidar_features = torch.load('lidar_features.pt',map_location="cpu")

logger.info("lidar_features={}",lidar_features)
logger.info("lidar_features.shape={}",lidar_features.shape)
logger.info("lidar_features.dtype={}",lidar_features.dtype)

codegen_py(model, lidar_features , export_path="adap_avgpool2d_c2")
