#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Script to precompute and save freqs_cis (rotary positional embeddings)
for DeepSeek models.

This script:
1. Computes the freqs_cis tensor using the model's precompute_freqs_cis function
2. Saves it as a complex tensor to DeepSeek_params/freqs_cis.pt
3. Demonstrates how to load and convert it to real-valued format
"""

import os
from pathlib import Path

import torch
from deepseek_v3p2_exp_model import ModelArgs, precompute_freqs_cis


def main():
    # Create output directory if it doesn't exist
    output_dir = Path("DeepSeek_params")
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Precomputing freqs_cis for DeepSeek models")
    print("=" * 60)

    # Define model configurations to precompute
    # Using the same configuration as deepseek_test.py
    batch_size = 2
    seq_len = 128

    configs = {
        "test_config": ModelArgs(
            max_batch_size=batch_size,
            max_seq_len=seq_len,
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
            index_n_heads=0,  # Disabled
            index_head_dim=32,  # Tile-aligned
            index_topk=0,  # Disabled
        ),
        "default": ModelArgs(),
    }

    for config_name, args in configs.items():
        print(f"\nProcessing configuration: {config_name}")
        print(f"  max_seq_len: {args.max_seq_len}")
        print(f"  qk_rope_head_dim: {args.qk_rope_head_dim}")
        print(f"  rope_theta: {args.rope_theta}")
        print(f"  rope_factor: {args.rope_factor}")

        # Compute freqs_cis (returns complex tensor)
        print("  Computing freqs_cis...")
        freqs_cis_complex = precompute_freqs_cis(args)

        print(f"  freqs_cis shape: {freqs_cis_complex.shape}")
        print(f"  freqs_cis dtype: {freqs_cis_complex.dtype}")

        # Save complex tensor
        output_path = output_dir / f"freqs_cis_{config_name}.pt"
        print(f"  Saving to {output_path}...")
        torch.save(freqs_cis_complex, output_path)

        # Verify we can load it back
        print(f"  Verifying saved file...")
        loaded_complex = torch.load(output_path)
        assert torch.allclose(
            freqs_cis_complex, loaded_complex
        ), "Loaded tensor doesn't match!"

        # Convert to real-valued tensor
        freqs_cis_real = torch.view_as_real(loaded_complex)
        print(f"  Converted to real format: {freqs_cis_real.shape}")
        print(f"  Real tensor dtype: {freqs_cis_real.dtype}")

        # Save real-valued version as well
        real_output_path = output_dir / f"freqs_cis_{config_name}_real.pt"
        print(f"  Saving real-valued version to {real_output_path}...")
        torch.save(freqs_cis_real, real_output_path)

    print("\n" + "=" * 60)
    print("Done! Files saved to DeepSeek_params/")
    print("=" * 60)

    # Demonstrate loading and usage
    print("\n" + "=" * 60)
    print("Example usage:")
    print("=" * 60)
    print(
        """
# Load complex format:
freqs_cis_complex = torch.load("DeepSeek_params/freqs_cis_default.pt")

# Convert complex tensor to real tensor with shape [..., 2],
# where the last dim is [real, imag]
freqs_cis_full = torch.view_as_real(freqs_cis_complex)

# Or load pre-converted real format directly:
freqs_cis_real = torch.load("DeepSeek_params/freqs_cis_default_real.pt")
"""
    )


if __name__ == "__main__":
    main()
