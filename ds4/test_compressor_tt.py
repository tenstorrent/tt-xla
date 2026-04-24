# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test simplified Compressor on TT device vs original Compressor on CPU.

This test:
1. Runs original Compressor.forward() on CPU → golden reference
2. Runs simplified compressor_forward_prefill() on TT device
3. Compares outputs with PCC

Run with: pytest -svv ds4/test_compressor_tt.py
"""
import sys
from pathlib import Path

# Add ds4 directory to path for imports
ds4_path = Path(__file__).parent
if str(ds4_path) not in sys.path:
    sys.path.insert(0, str(ds4_path))

# Install kernel stubs BEFORE importing from model.py
import kernel_stubs
kernel_stubs.install()

import pytest
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

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


def compute_pcc(a, b):
    """Compute Pearson Correlation Coefficient between two tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()

    a_mean = a_flat.mean()
    b_mean = b_flat.mean()

    a_centered = a_flat - a_mean
    b_centered = b_flat - b_mean

    numerator = (a_centered * b_centered).sum()
    denominator = torch.sqrt((a_centered ** 2).sum() * (b_centered ** 2).sum())

    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0

    return (numerator / denominator).item()


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [16, 32])
def test_compressor_simplified_tt_vs_original_cpu(batch_size, seq_len):
    """
    Compare simplified Compressor on TT vs original on CPU.

    - Original Compressor runs on CPU (golden reference)
    - Simplified compressor_forward_prefill runs on TT device
    """
    xr.set_device_type("TT")
    device = xm.xla_device()

    torch.manual_seed(42)

    # Small dimensions for fast testing
    args = ModelArgs()
    args.dim = 256
    args.head_dim = 64
    args.rope_head_dim = 16
    args.max_batch_size = batch_size
    args.max_seq_len = seq_len * 4

    compress_ratio = 4

    # Create compressor on CPU
    compressor_cpu = Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim)
    compressor_cpu.apply(init_weights)
    compressor_cpu.eval()

    # Setup buffers for CPU compressor
    compressor_cpu.kv_cache = torch.zeros(
        batch_size, args.max_seq_len // compress_ratio, args.head_dim,
        dtype=torch.bfloat16
    )
    compressor_cpu.freqs_cis = precompute_freqs_cis(
        args.rope_head_dim, args.max_seq_len, 0,
        args.rope_theta, args.rope_factor, args.beta_fast, args.beta_slow
    )

    # Input on CPU
    x_cpu = torch.randn(batch_size, seq_len, args.dim, dtype=torch.bfloat16)

    # Run ORIGINAL forward on CPU (golden reference)
    with torch.no_grad():
        out_cpu = compressor_cpu.forward(x_cpu, start_pos=0)

    print(f"\nCPU original output shape: {out_cpu.shape}")
    print(f"CPU output sample: {out_cpu[0, 0, :4]}")

    # Create compressor for TT with same weights
    compressor_tt = Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim)
    compressor_tt.load_state_dict(compressor_cpu.state_dict())
    compressor_tt.eval()

    # Move to TT device
    compressor_tt = compressor_tt.to(device)

    # Setup buffers on TT
    compressor_tt.kv_cache = torch.zeros(
        batch_size, args.max_seq_len // compress_ratio, args.head_dim,
        dtype=torch.bfloat16, device=device
    )
    compressor_tt.freqs_cis = precompute_freqs_cis(
        args.rope_head_dim, args.max_seq_len, 0,
        args.rope_theta, args.rope_factor, args.beta_fast, args.beta_slow
    ).to(device)

    # Input on TT
    x_tt = x_cpu.to(device)

    # Install simplified forward for TT
    kernel_stubs.install_compressor_prefill()

    # Run SIMPLIFIED forward on TT
    with torch.no_grad():
        out_tt = compressor_tt.forward(x_tt, start_pos=0)

    # Restore original forward
    kernel_stubs.uninstall_compressor_prefill()

    # Sync and move TT output to CPU for comparison
    xm.mark_step()
    out_tt_cpu = out_tt.cpu()

    print(f"TT simplified output shape: {out_tt_cpu.shape}")
    print(f"TT output sample: {out_tt_cpu[0, 0, :4]}")

    # Compare
    assert out_cpu.shape == out_tt_cpu.shape, \
        f"Shape mismatch: CPU {out_cpu.shape} vs TT {out_tt_cpu.shape}"

    pcc = compute_pcc(out_cpu, out_tt_cpu)
    max_diff = (out_cpu - out_tt_cpu).abs().max().item()
    mean_diff = (out_cpu - out_tt_cpu).abs().mean().item()

    print(f"PCC: {pcc:.6f}")
    print(f"Max diff: {max_diff}")
    print(f"Mean diff: {mean_diff}")

    assert pcc >= 0.95, f"PCC too low: {pcc} < 0.95"
    print(f"PASSED: batch_size={batch_size}, seq_len={seq_len}, PCC={pcc:.4f}")


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [128, 256])
def test_compressor_hsa_simplified_tt_vs_original_cpu(batch_size, seq_len):
    """
    Compare simplified HSA Compressor (ratio=128) on TT vs original on CPU.
    """
    xr.set_device_type("TT")
    device = xm.xla_device()

    torch.manual_seed(42)

    # Small dimensions for fast testing
    args = ModelArgs()
    args.dim = 256
    args.head_dim = 64
    args.rope_head_dim = 16
    args.max_batch_size = batch_size
    args.max_seq_len = seq_len * 2

    compress_ratio = 128

    # Create compressor on CPU
    compressor_cpu = Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim)
    compressor_cpu.apply(init_weights)
    compressor_cpu.eval()

    # Setup buffers for CPU compressor
    compressor_cpu.kv_cache = torch.zeros(
        batch_size, args.max_seq_len // compress_ratio, args.head_dim,
        dtype=torch.bfloat16
    )
    compressor_cpu.freqs_cis = precompute_freqs_cis(
        args.rope_head_dim, args.max_seq_len, 0,
        args.rope_theta, args.rope_factor, args.beta_fast, args.beta_slow
    )

    # Input on CPU
    x_cpu = torch.randn(batch_size, seq_len, args.dim, dtype=torch.bfloat16)

    # Run ORIGINAL forward on CPU (golden reference)
    with torch.no_grad():
        out_cpu = compressor_cpu.forward(x_cpu, start_pos=0)

    print(f"\nCPU original output shape: {out_cpu.shape}")

    # Create compressor for TT with same weights
    compressor_tt = Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim)
    compressor_tt.load_state_dict(compressor_cpu.state_dict())
    compressor_tt.eval()

    # Move to TT device
    compressor_tt = compressor_tt.to(device)

    # Setup buffers on TT
    compressor_tt.kv_cache = torch.zeros(
        batch_size, args.max_seq_len // compress_ratio, args.head_dim,
        dtype=torch.bfloat16, device=device
    )
    compressor_tt.freqs_cis = precompute_freqs_cis(
        args.rope_head_dim, args.max_seq_len, 0,
        args.rope_theta, args.rope_factor, args.beta_fast, args.beta_slow
    ).to(device)

    # Input on TT
    x_tt = x_cpu.to(device)

    # Install simplified forward for TT
    kernel_stubs.install_compressor_prefill()

    # Run SIMPLIFIED forward on TT
    with torch.no_grad():
        out_tt = compressor_tt.forward(x_tt, start_pos=0)

    # Restore original forward
    kernel_stubs.uninstall_compressor_prefill()

    # Sync and move TT output to CPU for comparison
    xm.mark_step()
    out_tt_cpu = out_tt.cpu()

    print(f"TT simplified output shape: {out_tt_cpu.shape}")

    # Compare
    assert out_cpu.shape == out_tt_cpu.shape, \
        f"Shape mismatch: CPU {out_cpu.shape} vs TT {out_tt_cpu.shape}"

    pcc = compute_pcc(out_cpu, out_tt_cpu)
    max_diff = (out_cpu - out_tt_cpu).abs().max().item()
    mean_diff = (out_cpu - out_tt_cpu).abs().mean().item()

    print(f"PCC: {pcc:.6f}")
    print(f"Max diff: {max_diff}")
    print(f"Mean diff: {mean_diff}")

    assert pcc >= 0.95, f"PCC too low: {pcc} < 0.95"
    print(f"PASSED: batch_size={batch_size}, seq_len={seq_len}, PCC={pcc:.4f}")


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("start_pos", [0, 1, 2])  # Positions that don't trigger compression (not 3)
def test_compressor_decode_accumulate_csa_tt_vs_cpu(batch_size, start_pos):
    """
    Compare decode accumulate (CSA, overlap=True) on TT vs CPU.

    This tests the path where we insert into state buffer but don't compress.
    For ratio=4 with overlap, compression happens at start_pos=3,7,11,...
    So start_pos=0,1,2 should just accumulate.
    """
    from kernel_stubs import compressor_forward_decode_accumulate_overlap

    xr.set_device_type("TT")
    device = xm.xla_device()

    torch.manual_seed(42)

    args = ModelArgs()
    args.dim = 256
    args.head_dim = 64
    args.rope_head_dim = 16
    args.max_batch_size = batch_size
    args.max_seq_len = 256

    compress_ratio = 4

    # Create compressor on CPU
    compressor_cpu = Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim)
    compressor_cpu.apply(init_weights)
    compressor_cpu.eval()

    # Setup buffers
    compressor_cpu.kv_cache = torch.zeros(
        batch_size, args.max_seq_len // compress_ratio, args.head_dim,
        dtype=torch.bfloat16
    )
    compressor_cpu.freqs_cis = precompute_freqs_cis(
        args.rope_head_dim, args.max_seq_len, 0,
        args.rope_theta, args.rope_factor, args.beta_fast, args.beta_slow
    )

    # Single token decode input
    x_cpu = torch.randn(batch_size, 1, args.dim, dtype=torch.bfloat16)

    # Run ORIGINAL forward on CPU (uses the full decode path)
    with torch.no_grad():
        out_cpu = compressor_cpu.forward(x_cpu, start_pos=start_pos)

    # Should return None (no compression at these positions)
    assert out_cpu is None, f"Expected None for start_pos={start_pos}, got {out_cpu}"

    # Capture CPU state after accumulate
    kv_state_cpu = compressor_cpu.kv_state[:batch_size].clone()
    score_state_cpu = compressor_cpu.score_state[:batch_size].clone()

    # Create compressor for TT with same weights
    compressor_tt = Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim)
    compressor_tt.load_state_dict(compressor_cpu.state_dict())
    compressor_tt.eval()

    # Reset state buffers for TT (they were modified during CPU forward)
    compressor_tt.kv_state.zero_()
    compressor_tt.score_state.fill_(float("-inf"))

    # Move to TT device
    compressor_tt = compressor_tt.to(device)

    # Setup buffers on TT
    compressor_tt.kv_cache = torch.zeros(
        batch_size, args.max_seq_len // compress_ratio, args.head_dim,
        dtype=torch.bfloat16, device=device
    )
    compressor_tt.freqs_cis = precompute_freqs_cis(
        args.rope_head_dim, args.max_seq_len, 0,
        args.rope_theta, args.rope_factor, args.beta_fast, args.beta_slow
    ).to(device)

    x_tt = x_cpu.to(device)

    # Run SIMPLIFIED decode accumulate on TT
    with torch.no_grad():
        out_tt = compressor_forward_decode_accumulate_overlap(compressor_tt, x_tt, start_pos=start_pos)

    # Should also return None
    assert out_tt is None, f"Expected None from TT, got {out_tt}"

    # Sync and compare state buffers
    xm.mark_step()
    kv_state_tt = compressor_tt.kv_state[:batch_size].cpu()
    score_state_tt = compressor_tt.score_state[:batch_size].cpu()

    # For overlap=True, slot_idx = ratio + (start_pos % ratio)
    slot_idx = compress_ratio + (start_pos % compress_ratio)

    # Compare the slot that was written
    kv_pcc = compute_pcc(kv_state_cpu[:, slot_idx], kv_state_tt[:, slot_idx])
    score_pcc = compute_pcc(score_state_cpu[:, slot_idx], score_state_tt[:, slot_idx])

    print(f"\nstart_pos={start_pos}, slot_idx={slot_idx}")
    print(f"KV state PCC: {kv_pcc:.6f}")
    print(f"Score state PCC: {score_pcc:.6f}")

    assert kv_pcc >= 0.99, f"KV state PCC too low: {kv_pcc}"
    assert score_pcc >= 0.99, f"Score state PCC too low: {score_pcc}"
    print(f"PASSED: batch_size={batch_size}, start_pos={start_pos}")


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("start_pos", [0, 1, 5, 10, 50])  # Positions that don't trigger compression
def test_compressor_decode_accumulate_hsa_tt_vs_cpu(batch_size, start_pos):
    """
    Compare decode accumulate (HSA, overlap=False) on TT vs CPU.

    For ratio=128 without overlap, compression happens at start_pos=127,255,...
    So positions 0-126 should just accumulate.
    """
    from kernel_stubs import compressor_forward_decode_accumulate_no_overlap

    xr.set_device_type("TT")
    device = xm.xla_device()

    torch.manual_seed(42)

    args = ModelArgs()
    args.dim = 256
    args.head_dim = 64
    args.rope_head_dim = 16
    args.max_batch_size = batch_size
    args.max_seq_len = 256

    compress_ratio = 128

    # Create compressor on CPU
    compressor_cpu = Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim)
    compressor_cpu.apply(init_weights)
    compressor_cpu.eval()

    # Setup buffers
    compressor_cpu.kv_cache = torch.zeros(
        batch_size, args.max_seq_len // compress_ratio, args.head_dim,
        dtype=torch.bfloat16
    )
    compressor_cpu.freqs_cis = precompute_freqs_cis(
        args.rope_head_dim, args.max_seq_len, 0,
        args.rope_theta, args.rope_factor, args.beta_fast, args.beta_slow
    )

    # Single token decode input
    x_cpu = torch.randn(batch_size, 1, args.dim, dtype=torch.bfloat16)

    # Run ORIGINAL forward on CPU
    with torch.no_grad():
        out_cpu = compressor_cpu.forward(x_cpu, start_pos=start_pos)

    assert out_cpu is None, f"Expected None for start_pos={start_pos}"

    # Capture CPU state
    kv_state_cpu = compressor_cpu.kv_state[:batch_size].clone()
    score_state_cpu = compressor_cpu.score_state[:batch_size].clone()

    # Create compressor for TT
    compressor_tt = Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim)
    compressor_tt.load_state_dict(compressor_cpu.state_dict())
    compressor_tt.eval()

    # Reset state buffers
    compressor_tt.kv_state.zero_()
    compressor_tt.score_state.fill_(float("-inf"))

    compressor_tt = compressor_tt.to(device)

    compressor_tt.kv_cache = torch.zeros(
        batch_size, args.max_seq_len // compress_ratio, args.head_dim,
        dtype=torch.bfloat16, device=device
    )
    compressor_tt.freqs_cis = precompute_freqs_cis(
        args.rope_head_dim, args.max_seq_len, 0,
        args.rope_theta, args.rope_factor, args.beta_fast, args.beta_slow
    ).to(device)

    x_tt = x_cpu.to(device)

    # Run SIMPLIFIED decode accumulate on TT
    with torch.no_grad():
        out_tt = compressor_forward_decode_accumulate_no_overlap(compressor_tt, x_tt, start_pos=start_pos)

    assert out_tt is None

    xm.mark_step()
    kv_state_tt = compressor_tt.kv_state[:batch_size].cpu()
    score_state_tt = compressor_tt.score_state[:batch_size].cpu()

    # For overlap=False, slot_idx = start_pos % ratio
    slot_idx = start_pos % compress_ratio

    kv_pcc = compute_pcc(kv_state_cpu[:, slot_idx], kv_state_tt[:, slot_idx])
    score_pcc = compute_pcc(score_state_cpu[:, slot_idx], score_state_tt[:, slot_idx])

    print(f"\nstart_pos={start_pos}, slot_idx={slot_idx}")
    print(f"KV state PCC: {kv_pcc:.6f}")
    print(f"Score state PCC: {score_pcc:.6f}")

    assert kv_pcc >= 0.99, f"KV state PCC too low: {kv_pcc}"
    assert score_pcc >= 0.99, f"Score state PCC too low: {score_pcc}"
    print(f"PASSED: batch_size={batch_size}, start_pos={start_pos}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing CSA Compressor Prefill (ratio=4)")
    print("=" * 60)
    test_compressor_simplified_tt_vs_original_cpu(batch_size=1, seq_len=32)

    print("\n" + "=" * 60)
    print("Testing HSA Compressor Prefill (ratio=128)")
    print("=" * 60)
    test_compressor_hsa_simplified_tt_vs_original_cpu(batch_size=1, seq_len=128)

    print("\n" + "=" * 60)
    print("Testing CSA Decode Accumulate (ratio=4, overlap=True)")
    print("=" * 60)
    test_compressor_decode_accumulate_csa_tt_vs_cpu(batch_size=1, start_pos=0)
    test_compressor_decode_accumulate_csa_tt_vs_cpu(batch_size=1, start_pos=1)
    test_compressor_decode_accumulate_csa_tt_vs_cpu(batch_size=1, start_pos=2)

    print("\n" + "=" * 60)
    print("Testing HSA Decode Accumulate (ratio=128, overlap=False)")
    print("=" * 60)
    test_compressor_decode_accumulate_hsa_tt_vs_cpu(batch_size=1, start_pos=0)
    test_compressor_decode_accumulate_hsa_tt_vs_cpu(batch_size=1, start_pos=10)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
