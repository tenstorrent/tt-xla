# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn
import torch_xla.runtime as xr
from tt_torch import codegen_py


# Set up XLA runtime for TT backend.
xr.set_device_type("TT")

class concat(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4 ):

        return torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)


model = concat()

pred_boxes1 = torch.randn(217413, 1, dtype=torch.float32)
pred_boxes2 = torch.randn(217413, 1, dtype=torch.float32)
pred_boxes3 = torch.randn(217413, 1, dtype=torch.float32)
pred_boxes4 = torch.randn(217413, 1, dtype=torch.float32)


x = [pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4]

codegen_py(model, *x, export_path="concat",export_tensors=False)
