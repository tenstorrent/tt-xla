# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch_xla
import torch_xla.runtime as xr
from deepseek_v3p2_exp_model import ModelArgs, Transformer


# --------------------------------
# Test run
# --------------------------------
def deepseek_test():
    # Set device type
    xr.set_device_type("TT")

    # Create model args with a single layer for testing
    args = ModelArgs(
        max_batch_size=1,
        max_seq_len=32,
        dtype="bf16",
        vocab_size=4096,  # 128 * 32 - tile-aligned
        dim=512,  # 16 * 32 - reduced from 2048
        inter_dim=1024,  # 32 * 32 - reduced from 10944, tile-aligned
        moe_inter_dim=512,  # 16 * 32 - reduced from 1408, tile-aligned
        n_layers=1,
        n_dense_layers=1,  # Dense layer only, no MoE
        n_heads=8,  # Reduced from 16, must divide dim evenly
        n_routed_experts=8,  # Reduced (won't be used since n_dense_layers=1)
        n_shared_experts=2,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        score_func="softmax",
        route_scale=1.0,
        q_lora_rank=128,  # Changed from 0 - MUST be non-zero and tile-aligned
        kv_lora_rank=256,  # 8 * 32 - reduced from 512, tile-aligned
        qk_nope_head_dim=32,  # 1 * 32 - reduced from 128, tile-aligned
        qk_rope_head_dim=32,  # 1 * 32 - reduced from 64, tile-aligned
        v_head_dim=64,  # 2 * 32 - reduced from 128, tile-aligned
        original_seq_len=4096,
        rope_theta=10000.0,
        rope_factor=40,
        beta_fast=32,
        beta_slow=1,
        mscale=1.0,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=2048,
    )

    # Instantiate model with precomputed freqs_cis
    # Option 1: Load from file (recommended for performance)
    freqs_cis_path = "/localdev/gengelage/tt-xla/examples/pytorch/DeepSeek_params/freqs_cis_test_config_real.pt"
    model = Transformer(args, freqs_cis_path=freqs_cis_path)

    # Option 2: Compute on the fly (fallback)
    # model = Transformer(args)

    model = model.to(torch.bfloat16)

    # Put it in inference mode and compile it
    model = model.eval()
    compiled_model = torch.compile(model, backend="tt")

    # Create batch of random token IDs
    batch_size = 1
    seq_len = 32
    tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len))

    # Move inputs and model to device
    device = torch_xla.device()
    tokens = tokens.to(device)
    compiled_model = compiled_model.to(device)

    # Run model
    with torch.no_grad():
        output = compiled_model(tokens)


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    deepseek_test()
