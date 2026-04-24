# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Validate simplified Compressor produces same results as original on CPU.

This test compares the original Compressor.forward() against the simplified
compressor_forward_prefill() to ensure they produce identical outputs when
seqlen is divisible by compress_ratio.

Run with: python ds4/test_compressor_validation.py
"""
import sys
from pathlib import Path

ds4_path = Path(__file__).parent
if str(ds4_path) not in sys.path:
    sys.path.insert(0, str(ds4_path))

# Install kernel stubs BEFORE importing from model.py
import kernel_stubs
kernel_stubs.install()

import torch
import torch.nn as nn
from model import ModelArgs, Compressor, precompute_freqs_cis


def init_weights(module):
    """Initialize weights with random values to avoid NaN/inf from uninitialized memory."""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.zeros_(module.bias)
    # Initialize Compressor's APE (Absolute Positional Encoding) parameter
    if hasattr(module, 'ape') and module.ape is not None:
        nn.init.normal_(module.ape, mean=0.0, std=0.02)


def test_compressor_prefill_equivalence_csa():
    """Compare original vs simplified Compressor on CPU for CSA (ratio=4)."""
    print("=" * 60)
    print("Testing CSA Compressor (compress_ratio=4, overlap=True)")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 2
    seq_len = 32  # Must be divisible by compress_ratio=4
    compress_ratio = 4

    args = ModelArgs()
    args.dim = 256
    args.head_dim = 64
    args.rope_head_dim = 16
    args.max_batch_size = batch_size
    args.max_seq_len = seq_len * 4

    # Create and init compressor
    compressor = Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim)
    compressor.apply(init_weights)
    compressor.eval()

    # Setup buffers
    compressor.kv_cache = torch.zeros(
        batch_size, args.max_seq_len // compress_ratio, args.head_dim,
        dtype=torch.bfloat16
    )
    compressor.freqs_cis = precompute_freqs_cis(
        args.rope_head_dim, args.max_seq_len, 0,
        args.rope_theta, args.rope_factor, args.beta_fast, args.beta_slow
    )

    # Input
    x = torch.randn(batch_size, seq_len, args.dim, dtype=torch.bfloat16)

    # Run original forward
    with torch.no_grad():
        out_original = compressor.forward(x, start_pos=0)

    print(f"Original output shape: {out_original.shape}")

    # Install simplified forward and run
    kernel_stubs.install_compressor_prefill()

    with torch.no_grad():
        out_simplified = compressor.forward(x, start_pos=0)

    # Restore original
    kernel_stubs.uninstall_compressor_prefill()

    print(f"Simplified output shape: {out_simplified.shape}")

    # Compare
    assert out_original.shape == out_simplified.shape, \
        f"Shape mismatch: {out_original.shape} vs {out_simplified.shape}"

    diff = (out_original - out_simplified).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max diff: {max_diff}")
    print(f"Mean diff: {mean_diff}")

    # Should be identical (or very close due to fp precision)
    assert max_diff < 1e-5, f"Outputs differ too much: max_diff={max_diff}"
    print("PASSED: CSA Simplified Compressor matches original!\n")


def test_compressor_prefill_equivalence_hsa():
    """Compare original vs simplified Compressor on CPU for HSA (ratio=128)."""
    print("=" * 60)
    print("Testing HSA Compressor (compress_ratio=128, overlap=False)")
    print("=" * 60)

    torch.manual_seed(42)

    batch_size = 2
    seq_len = 256  # Must be divisible by compress_ratio=128
    compress_ratio = 128

    args = ModelArgs()
    args.dim = 256
    args.head_dim = 64
    args.rope_head_dim = 16
    args.max_batch_size = batch_size
    args.max_seq_len = seq_len * 2

    # Create and init compressor
    compressor = Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim)
    compressor.apply(init_weights)
    compressor.eval()

    # Setup buffers
    compressor.kv_cache = torch.zeros(
        batch_size, args.max_seq_len // compress_ratio, args.head_dim,
        dtype=torch.bfloat16
    )
    compressor.freqs_cis = precompute_freqs_cis(
        args.rope_head_dim, args.max_seq_len, 0,
        args.rope_theta, args.rope_factor, args.beta_fast, args.beta_slow
    )

    # Input
    x = torch.randn(batch_size, seq_len, args.dim, dtype=torch.bfloat16)

    # Run original forward
    with torch.no_grad():
        out_original = compressor.forward(x, start_pos=0)

    print(f"Original output shape: {out_original.shape}")

    # Install simplified forward and run
    kernel_stubs.install_compressor_prefill()

    with torch.no_grad():
        out_simplified = compressor.forward(x, start_pos=0)

    # Restore original
    kernel_stubs.uninstall_compressor_prefill()

    print(f"Simplified output shape: {out_simplified.shape}")

    # Compare
    assert out_original.shape == out_simplified.shape, \
        f"Shape mismatch: {out_original.shape} vs {out_simplified.shape}"

    diff = (out_original - out_simplified).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max diff: {max_diff}")
    print(f"Mean diff: {mean_diff}")

    # Should be identical (or very close due to fp precision)
    assert max_diff < 1e-5, f"Outputs differ too much: max_diff={max_diff}"
    print("PASSED: HSA Simplified Compressor matches original!\n")


if __name__ == "__main__":
    test_compressor_prefill_equivalence_csa()
    test_compressor_prefill_equivalence_hsa()
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
