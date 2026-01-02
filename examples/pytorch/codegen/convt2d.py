# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch_xla.runtime as xr
from loguru import logger
from tt_torch import codegen_py

# Set up XLA runtime for TT backend
xr.set_device_type("TT")


class ConvTranspose2dModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(
            2048, 128, kernel_size=(2, 2), stride=(2, 2), bias=False
        )

    def forward(self, input_tensor):
        return self.conv_transpose(input_tensor)


model = ConvTranspose2dModel()
model.eval()
logger.info("model={}", model)
x = torch.randn(6, 2048, 8, 22)

codegen_py(model, x, export_path="convt2d_emitpy", export_tensors=False)
