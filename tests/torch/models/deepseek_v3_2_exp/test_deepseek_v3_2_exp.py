# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from modeling_deepseek_v3_2_exp import ModelArgs, Transformer

# This model is modified from the original deepseek_v3_2_exp model.py to:
# 1. Use scipy.linalg.hadamard instead of fast_hadamard_transform
# 2. Disable FP8 quantization features (act_quant, fp8_gemm, fp8_index) with stubs
# 3. Avoid torch.view_as_complex/view_as_real operations


@pytest.mark.xfail(
    reason="TT_THROW: Statically allocated circular buffers on core range [(x=7,y=6) - (x=7,y=6)] grow to 16897152 B which is beyond max L1 size of 1499136 B"
)
def test_deepseek_transformer_single_layer():
    xr.set_device_type("TT")

    # Create model args with a single layer for testing
    args = ModelArgs(
        n_layers=1,
        q_lora_rank=3072,
    )

    model = Transformer(args)

    model = model.to(torch.bfloat16)

    model = model.eval()
    compiled_model = torch.compile(model, backend="tt")

    batch_size = 1
    seq_len = 32
    tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len))

    device = torch_xla.device()
    tokens = tokens.to(device)
    compiled_model = compiled_model.to(device)

    with torch.no_grad():
        output = compiled_model(tokens)
        output.to("cpu")
