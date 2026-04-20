# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch_xla.runtime as xr
from transformers import ResNetForImageClassification
from tt_torch import codegen_py
from loguru import logger

# Set up XLA runtime for TT backend
xr.set_device_type("TT")


class SDPA(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, attn_bias):
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_bias,
            dropout_p=0.0,
            is_causal=False,
            scale=0.125,
        )



model = SDPA().to(torch.bfloat16)
model.eval()

l1_query_layer = torch.load("l1_query_layer.pt",map_location="cpu")
l1_key_layer = torch.load("l1_key_layer.pt",map_location="cpu")
l1_value_layer = torch.load("l1_value_layer.pt",map_location="cpu")
l1_attn_bias = torch.load("l1_attn_bias.pt",map_location="cpu")

logger.info("l1_query_layer={}",l1_query_layer)
logger.info("l1_key_layer={}",l1_key_layer)
logger.info("l1_value_layer={}",l1_value_layer)
logger.info("l1_attn_bias={}",l1_attn_bias)

logger.info("l1_query_layer.shape={}",l1_query_layer.shape)
logger.info("l1_key_layer.shape={}",l1_key_layer.shape)
logger.info("l1_value_layer.shape={}",l1_value_layer.shape)
logger.info("l1_attn_bias.shape={}",l1_attn_bias.shape)

logger.info("l1_query_layer.dtype={}",l1_query_layer.dtype)
logger.info("l1_key_layer.dtype={}",l1_key_layer.dtype)
logger.info("l1_value_layer.dtype={}",l1_value_layer.dtype)
logger.info("l1_attn_bias.dtype={}",l1_attn_bias.dtype)

inputs = [l1_query_layer, l1_key_layer, l1_value_layer, l1_attn_bias]
    

codegen_py(model, *inputs, export_path="sdpa_beit_base")
