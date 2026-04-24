# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TT Compressor prefill tests with padded sequence lengths.

Tests the simplified compressor_forward_prefill() on TT device vs original
Compressor.forward() on CPU, comparing outputs with PCC.

Run with: pytest -svv ds4/test_compressor_prefill_tt.py
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
from kernel_stubs import compressor_forward_prefill

torch.set_default_dtype(torch.bfloat16)


def init_weights(module):
    """Initialize weights properly for testing."""
    for name, param in module.named_parameters():
        if param.requires_grad:
            with torch.no_grad():
                if 'weight' in name and param.dim() >= 2:
                    init_data = torch.empty_like(param, dtype=torch.float32)
                    nn.init.xavier_uniform_(init_data)
                    param.copy_(init_data.to(param.dtype))
                elif 'bias' in name:
                    param.zero_()
                elif param.dim() == 1:
                    param.fill_(1.0)


def compute_pcc(a, b):
    """Compute Pearson Correlation Coefficient between two tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    a_centered = a_flat - a_flat.mean()
    b_centered = b_flat - b_flat.mean()
    num = (a_centered * b_centered).sum()
    denom = torch.sqrt((a_centered ** 2).sum() * (b_centered ** 2).sum())
    if denom < 1e-8:
        # If variance is near zero, check if values are close
        return 1.0 if torch.allclose(a_flat, b_flat, rtol=1e-2, atol=1e-2) else 0.0
    return (num / denom).item()


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [32, 64, 128])
def test_compressor_prefill_csa_tt_vs_cpu(batch_size, seq_len):
    """
    Test CSA Compressor (ratio=4) prefill on TT vs CPU.

    Uses simplified compressor_forward_prefill() which is statically compilable.
    Sequence length must be divisible by ratio.
    """
    xr.set_device_type("TT")
    device = xm.xla_device()
    torch.manual_seed(42)

    compress_ratio = 4
    assert seq_len % compress_ratio == 0, f"seq_len must be divisible by {compress_ratio}"

    args = ModelArgs(
        max_batch_size=batch_size,
        max_seq_len=512,
        dim=256,
        head_dim=64,
        rope_head_dim=16,
    )

    # Create compressor on CPU
    comp_cpu = Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim)
    init_weights(comp_cpu)
    comp_cpu.eval()

    # Setup buffers
    comp_cpu.kv_cache = torch.zeros(
        batch_size, args.max_seq_len // compress_ratio, args.head_dim,
        dtype=torch.bfloat16
    )
    comp_cpu.freqs_cis = precompute_freqs_cis(
        args.rope_head_dim, args.max_seq_len, 0,
        args.rope_theta, args.rope_factor, args.beta_fast, args.beta_slow
    )

    # Input
    x_cpu = torch.randn(batch_size, seq_len, args.dim, dtype=torch.bfloat16)

    # Run ORIGINAL forward on CPU (golden reference)
    with torch.no_grad():
        out_cpu_original = comp_cpu.forward(x_cpu, start_pos=0)

    # Run SIMPLIFIED forward on CPU (should match original)
    comp_cpu.kv_cache.zero_()
    with torch.no_grad():
        out_cpu_simplified = compressor_forward_prefill(comp_cpu, x_cpu)

    # Verify CPU original == CPU simplified
    pcc_cpu = compute_pcc(out_cpu_original, out_cpu_simplified)
    assert pcc_cpu >= 0.999, f"CPU original vs simplified PCC too low: {pcc_cpu}"

    # Create compressor for TT with same weights
    comp_tt = Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim)
    comp_tt.load_state_dict(comp_cpu.state_dict())
    comp_tt.eval()
    comp_tt = comp_tt.to(device)

    # Setup buffers on TT
    comp_tt.kv_cache = torch.zeros(
        batch_size, args.max_seq_len // compress_ratio, args.head_dim,
        dtype=torch.bfloat16, device=device
    )
    comp_tt.freqs_cis = precompute_freqs_cis(
        args.rope_head_dim, args.max_seq_len, 0,
        args.rope_theta, args.rope_factor, args.beta_fast, args.beta_slow
    ).to(device)

    x_tt = x_cpu.to(device)

    # Run SIMPLIFIED forward on TT
    with torch.no_grad():
        out_tt = compressor_forward_prefill(comp_tt, x_tt)

    xm.mark_step()
    out_tt_cpu = out_tt.cpu()

    # Compare
    assert out_cpu_simplified.shape == out_tt_cpu.shape
    assert not torch.isnan(out_tt_cpu).any(), "NaN in TT output"

    pcc = compute_pcc(out_cpu_simplified, out_tt_cpu)
    max_diff = (out_cpu_simplified - out_tt_cpu).abs().max().item()

    print(f"\nbatch={batch_size}, seq_len={seq_len}")
    print(f"  PCC: {pcc:.6f}, Max diff: {max_diff:.4f}")

    assert pcc >= 0.95, f"PCC too low: {pcc} < 0.95"


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [128, 256])
def test_compressor_prefill_hsa_tt_vs_cpu(batch_size, seq_len):
    """
    Test HSA Compressor (ratio=128) prefill on TT vs CPU.

    HSA has no overlap, simpler compression.
    Sequence length must be divisible by ratio.
    """
    xr.set_device_type("TT")
    device = xm.xla_device()
    torch.manual_seed(42)

    compress_ratio = 128
    assert seq_len % compress_ratio == 0, f"seq_len must be divisible by {compress_ratio}"

    args = ModelArgs(
        max_batch_size=batch_size,
        max_seq_len=512,
        dim=256,
        head_dim=64,
        rope_head_dim=16,
    )

    # Create compressor on CPU
    comp_cpu = Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim)
    init_weights(comp_cpu)
    comp_cpu.eval()

    comp_cpu.kv_cache = torch.zeros(
        batch_size, args.max_seq_len // compress_ratio, args.head_dim,
        dtype=torch.bfloat16
    )
    comp_cpu.freqs_cis = precompute_freqs_cis(
        args.rope_head_dim, args.max_seq_len, 0,
        args.rope_theta, args.rope_factor, args.beta_fast, args.beta_slow
    )

    x_cpu = torch.randn(batch_size, seq_len, args.dim, dtype=torch.bfloat16)

    # Run on CPU
    with torch.no_grad():
        out_cpu = comp_cpu.forward(x_cpu, start_pos=0)

    # Create for TT
    comp_tt = Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim)
    comp_tt.load_state_dict(comp_cpu.state_dict())
    comp_tt.eval()
    comp_tt = comp_tt.to(device)

    comp_tt.kv_cache = torch.zeros(
        batch_size, args.max_seq_len // compress_ratio, args.head_dim,
        dtype=torch.bfloat16, device=device
    )
    comp_tt.freqs_cis = precompute_freqs_cis(
        args.rope_head_dim, args.max_seq_len, 0,
        args.rope_theta, args.rope_factor, args.beta_fast, args.beta_slow
    ).to(device)

    x_tt = x_cpu.to(device)

    # Run simplified on TT
    with torch.no_grad():
        out_tt = compressor_forward_prefill(comp_tt, x_tt)

    xm.mark_step()
    out_tt_cpu = out_tt.cpu()

    assert out_cpu.shape == out_tt_cpu.shape
    assert not torch.isnan(out_tt_cpu).any()

    pcc = compute_pcc(out_cpu, out_tt_cpu)
    max_diff = (out_cpu - out_tt_cpu).abs().max().item()
    print(f"\nHSA batch={batch_size}, seq_len={seq_len}: PCC={pcc:.6f}, Max diff={max_diff:.4f}")

    # HSA has ratio=128, so longer sequences pool more tokens and accumulate more error
    # Use lower threshold for HSA
    assert pcc >= 0.90, f"PCC too low: {pcc}"


if __name__ == "__main__":
    print("=" * 60)
    print("Testing CSA Compressor Prefill (ratio=4)")
    print("=" * 60)
    test_compressor_prefill_csa_tt_vs_cpu(batch_size=1, seq_len=128)

    print("\n" + "=" * 60)
    print("Testing HSA Compressor Prefill (ratio=128)")
    print("=" * 60)
    test_compressor_prefill_hsa_tt_vs_cpu(batch_size=1, seq_len=128)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
