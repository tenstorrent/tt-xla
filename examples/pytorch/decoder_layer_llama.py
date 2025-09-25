# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
)
from transformers.cache_utils import StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast
import os
import numpy as np
from torch_xla.distributed.spmd import Mesh
import torch_xla.distributed.spmd as xs
import transformers.models.llama.modeling_llama as llama_model



# --------------------------------
# Llama Generation Example
# --------------------------------
def llama():
    # Connect the device.
    device = xm.xla_device()

    # Instantiate model.
    model_name: str = "meta-llama/Llama-3.2-3B"

    model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, use_cache=True
    )

    decoder_layer_model: torch.nn.Module = llama_model.LlamaDecoderLayer(model.config, 0)

    # Put it in inference mode
    decoder_layer_model = decoder_layer_model.eval()

    with torch.no_grad():
        # Instantiate static cache on host then transfer it to device to avoid CE creation ops
        batch_size = 1
        max_cache_len = 16
        static_cache: StaticCache = StaticCache(
            config=model.config,
            max_batch_size=batch_size,
            max_cache_len=max_cache_len,
            device="cpu",
            # device='xla',  # 'xla' device will create the cache on host then we move it to device
            dtype=torch.bfloat16,
        )

        static_cache.key_cache = [k.to(device) for k in static_cache.key_cache]
        static_cache.value_cache = [v.to(device) for v in static_cache.value_cache]

    # Move inputs and model to device.
    decoder_layer_model = decoder_layer_model.to(device)

    seq_len = 1
    head_dim = model.config.hidden_size // model.config.num_attention_heads  # 128
    print(f"Using head dim of {head_dim}")

    hidden_states = torch.randn(batch_size, seq_len, model.config.hidden_size).to(device)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(device)

    # Position embeddings should be [batch, seq_len, head_dim] for rotary
    position_embeddings = (
        torch.randn(batch_size, seq_len, head_dim).to(device),  # cos
        torch.randn(batch_size, seq_len, head_dim).to(device)   # sin
    )
    cache_position = torch.arange(seq_len).to(device)

    decoder_layer_model.compile(backend="tt")

    with torch.no_grad():
        output = decoder_layer_model(hidden_states, position_ids=position_ids, position_embeddings=position_embeddings, cache_position=cache_position, past_key_value=static_cache, use_cache=True)

    print(output)

# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    llama()

