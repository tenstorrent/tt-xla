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
            n_layers=1,
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
