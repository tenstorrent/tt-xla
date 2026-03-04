# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch_xla.runtime as xr
from tt_torch import codegen_py
from loguru import logger
import torch.nn.functional as F

# Set up XLA runtime for TT backend
xr.set_device_type("TT")

class sanity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.score_thresh = 0.01

    def forward(self, cls_logits ):
        
        pred_scores = F.softmax(cls_logits, dim=-1)
        
        for scores in pred_scores:

            for label in range(1, 2):
                score = scores[:, label]
                keep_idxs = score > self.score_thresh
            
        return keep_idxs


model = sanity()
model.eval()

cls_logits = torch.load('cls_logits_org.pt',map_location="cpu")
logger.info("cls_logits={}",cls_logits)
logger.info("cls_logits.shape={}",cls_logits.shape)
logger.info("cls_logits.dtype={}",cls_logits.dtype)

codegen_py(model, cls_logits, export_path="sanity")

