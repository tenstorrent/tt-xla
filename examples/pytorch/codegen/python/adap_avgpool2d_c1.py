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
            self.avgpool_img = nn.AdaptiveAvgPool2d((5, 22))

        def forward(self, img_features ):
            
            output = self.avgpool_img(img_features)

            return output


model = adaptive_avgpool2d().to(torch.bfloat16)

image_features = torch.load('image_features.pt',map_location="cpu")

logger.info("image_features={}",image_features)
logger.info("image_features.shape={}",image_features.shape)
logger.info("image_features.dtype={}",image_features.dtype)

codegen_py(model, image_features , export_path="adap_avgpool2d_c1")
