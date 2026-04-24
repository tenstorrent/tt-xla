# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test RatioPadCompressor stub on TT device vs original Compressor on CPU.

This test verifies that ratio_pad_compressor_forward (the stub) produces the
same results as the original Compressor.forward.

Test structure:
1. Run original Compressor.forward() on CPU → golden reference
2. Run ratio_pad_compressor_forward() stub on TT device
3. Compare outputs with PCC

Run with: pytest -svv ds4/test_ratio_pad_compressor_tt.py
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
from kernel_stubs import ratio_pad_compressor_forward


def init_weights(module):
    """Initialize weights with random values to avoid NaN/inf from uninitialized memory."""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.zeros_(module.bias)
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


def setup_compressor(args, compress_ratio, batch_size, device=None):
    """Create and initialize a Compressor with proper buffers."""
    compressor = Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim)
    compressor.apply(init_weights)
    compressor.eval()

    if device is not None:
        compressor = compressor.to(device)

    # Setup buffers
    compressor.kv_cache = torch.zeros(
        batch_size, args.max_seq_len // compress_ratio, args.head_dim,
        dtype=torch.bfloat16, device=device
    )
    compressor.freqs_cis = precompute_freqs_cis(
        args.rope_head_dim, args.max_seq_len, 0,
        args.rope_theta, args.rope_factor, args.beta_fast, args.beta_slow
    )
    if device is not None:
        compressor.freqs_cis = compressor.freqs_cis.to(device)

    return compressor


# ============================================================================
# Prefill Tests - verify padding produces same results as original
# ============================================================================

@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [16, 32, 64])
def test_ratio_pad_compressor_prefill_csa_tt_vs_cpu(batch_size, seq_len):
    """
    Test CSA (ratio=4) prefill: stub on TT vs original Compressor.forward on CPU.
    """
    xr.set_device_type("TT")
    device = xm.xla_device()
    torch.manual_seed(42)

    args = ModelArgs()
    args.dim = 256
    args.head_dim = 64
    args.rope_head_dim = 16
    args.max_batch_size = batch_size
    args.max_seq_len = seq_len * 4

    compress_ratio = 4

    # CPU: original Compressor.forward (golden reference)
    compressor_cpu = setup_compressor(args, compress_ratio, batch_size)
    x_cpu = torch.randn(batch_size, seq_len, args.dim, dtype=torch.bfloat16)

    with torch.no_grad():
        out_cpu = compressor_cpu.forward(x_cpu, start_pos=0)

    print(f"\nCPU original output shape: {out_cpu.shape}")

    # TT: ratio_pad_compressor_forward stub
    compressor_tt = Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim)
    compressor_tt.load_state_dict(compressor_cpu.state_dict())
    compressor_tt.eval()
    compressor_tt = compressor_tt.to(device)

    compressor_tt.kv_cache = torch.zeros(
        batch_size, args.max_seq_len // compress_ratio, args.head_dim,
        dtype=torch.bfloat16, device=device
    )
    compressor_tt.freqs_cis = compressor_cpu.freqs_cis.to(device)

    x_tt = x_cpu.to(device)

    with torch.no_grad():
        out_tt = ratio_pad_compressor_forward(compressor_tt, x_tt, start_pos=0)

    xm.mark_step()
    out_tt_cpu = out_tt.cpu()

    print(f"TT stub output shape: {out_tt_cpu.shape}")

    assert out_cpu.shape == out_tt_cpu.shape, \
        f"Shape mismatch: CPU {out_cpu.shape} vs TT {out_tt_cpu.shape}"

    pcc = compute_pcc(out_cpu, out_tt_cpu)
    max_diff = (out_cpu - out_tt_cpu).abs().max().item()

    print(f"PCC: {pcc:.6f}, Max diff: {max_diff}")
    assert pcc >= 0.95, f"PCC too low: {pcc} < 0.95"
    print(f"PASSED: CSA prefill batch_size={batch_size}, seq_len={seq_len}")


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [128, 256])
def test_ratio_pad_compressor_prefill_hsa_tt_vs_cpu(batch_size, seq_len):
    """
    Test HSA (ratio=128) prefill: stub on TT vs original on CPU.
    """
    xr.set_device_type("TT")
    device = xm.xla_device()
    torch.manual_seed(42)

    args = ModelArgs()
    args.dim = 256
    args.head_dim = 64
    args.rope_head_dim = 16
    args.max_batch_size = batch_size
    args.max_seq_len = seq_len * 2

    compress_ratio = 128

    # CPU: original (golden)
    compressor_cpu = setup_compressor(args, compress_ratio, batch_size)
    x_cpu = torch.randn(batch_size, seq_len, args.dim, dtype=torch.bfloat16)

    with torch.no_grad():
        out_cpu = compressor_cpu.forward(x_cpu, start_pos=0)

    print(f"\nCPU original output shape: {out_cpu.shape}")

    # TT: stub
    compressor_tt = Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim)
    compressor_tt.load_state_dict(compressor_cpu.state_dict())
    compressor_tt.eval()
    compressor_tt = compressor_tt.to(device)

    compressor_tt.kv_cache = torch.zeros(
        batch_size, args.max_seq_len // compress_ratio, args.head_dim,
        dtype=torch.bfloat16, device=device
    )
    compressor_tt.freqs_cis = compressor_cpu.freqs_cis.to(device)

    x_tt = x_cpu.to(device)

    with torch.no_grad():
        out_tt = ratio_pad_compressor_forward(compressor_tt, x_tt, start_pos=0)

    xm.mark_step()
    out_tt_cpu = out_tt.cpu()

    print(f"TT stub output shape: {out_tt_cpu.shape}")

    assert out_cpu.shape == out_tt_cpu.shape
    pcc = compute_pcc(out_cpu, out_tt_cpu)
    print(f"PCC: {pcc:.6f}")
    assert pcc >= 0.95, f"PCC too low: {pcc}"
    print(f"PASSED: HSA prefill batch_size={batch_size}, seq_len={seq_len}")


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [17, 30, 65])  # Non-multiples of 4 to test padding
def test_ratio_pad_compressor_prefill_with_remainder_tt_vs_cpu(batch_size, seq_len):
    """
    Test CSA prefill with seq_len NOT divisible by ratio.

    This specifically tests the padding logic where:
    - Original Compressor splits and stores remainder in state buffer
    - ratio_pad_compressor_forward pads to multiple of ratio and masks with -inf
    """
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
    remainder = seq_len % compress_ratio
    expected_output_len = seq_len // compress_ratio

    print(f"\nseq_len={seq_len}, remainder={remainder}, expected_output_len={expected_output_len}")

    # CPU: original (golden)
    compressor_cpu = setup_compressor(args, compress_ratio, batch_size)
    x_cpu = torch.randn(batch_size, seq_len, args.dim, dtype=torch.bfloat16)

    with torch.no_grad():
        out_cpu = compressor_cpu.forward(x_cpu, start_pos=0)

    print(f"CPU original output shape: {out_cpu.shape}")
    assert out_cpu.shape[1] == expected_output_len, \
        f"CPU output length {out_cpu.shape[1]} != expected {expected_output_len}"

    # TT: stub
    compressor_tt = Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim)
    compressor_tt.load_state_dict(compressor_cpu.state_dict())
    compressor_tt.eval()
    compressor_tt = compressor_tt.to(device)

    compressor_tt.kv_cache = torch.zeros(
        batch_size, args.max_seq_len // compress_ratio, args.head_dim,
        dtype=torch.bfloat16, device=device
    )
    compressor_tt.freqs_cis = compressor_cpu.freqs_cis.to(device)

    x_tt = x_cpu.to(device)

    with torch.no_grad():
        out_tt = ratio_pad_compressor_forward(compressor_tt, x_tt, start_pos=0)

    xm.mark_step()
    out_tt_cpu = out_tt.cpu()

    print(f"TT stub output shape: {out_tt_cpu.shape}")

    assert out_cpu.shape == out_tt_cpu.shape, \
        f"Shape mismatch: CPU {out_cpu.shape} vs TT {out_tt_cpu.shape}"

    pcc = compute_pcc(out_cpu, out_tt_cpu)
    print(f"PCC: {pcc:.6f}")
    assert pcc >= 0.95, f"PCC too low: {pcc}"
    print(f"PASSED: CSA prefill with remainder, batch_size={batch_size}, seq_len={seq_len}")


# ============================================================================
# Decode Tests - verify decode path works correctly
# ============================================================================

@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("start_pos", [0, 1, 2])  # Accumulate only (compression at 3)
def test_ratio_pad_compressor_decode_accumulate_csa_tt_vs_cpu(batch_size, start_pos):
    """
    Test CSA decode accumulate: stub on TT vs original on CPU.

    For ratio=4, compression happens at start_pos=3,7,11,...
    Positions 0,1,2 should only accumulate into state buffer.
    """
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

    # CPU: original (golden)
    compressor_cpu = setup_compressor(args, compress_ratio, batch_size)
    x_cpu = torch.randn(batch_size, 1, args.dim, dtype=torch.bfloat16)

    with torch.no_grad():
        out_cpu = compressor_cpu.forward(x_cpu, start_pos=start_pos)

    # Should return None (accumulate only)
    assert out_cpu is None, f"Expected None for start_pos={start_pos}, got {out_cpu}"

    # Capture state
    kv_state_cpu = compressor_cpu.kv_state[:batch_size].clone()
    score_state_cpu = compressor_cpu.score_state[:batch_size].clone()

    # TT: stub
    compressor_tt = Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim)
    compressor_tt.load_state_dict(compressor_cpu.state_dict())
    compressor_tt.eval()

    # Reset state buffers before TT run
    compressor_tt.kv_state.zero_()
    compressor_tt.score_state.fill_(float("-inf"))

    compressor_tt = compressor_tt.to(device)

    compressor_tt.kv_cache = torch.zeros(
        batch_size, args.max_seq_len // compress_ratio, args.head_dim,
        dtype=torch.bfloat16, device=device
    )
    compressor_tt.freqs_cis = compressor_cpu.freqs_cis.to(device)

    x_tt = x_cpu.to(device)

    with torch.no_grad():
        out_tt = ratio_pad_compressor_forward(compressor_tt, x_tt, start_pos=start_pos)

    assert out_tt is None, f"Expected None from TT stub, got {out_tt}"

    xm.mark_step()
    kv_state_tt = compressor_tt.kv_state[:batch_size].cpu()
    score_state_tt = compressor_tt.score_state[:batch_size].cpu()

    # For overlap=True, slot_idx = ratio + (start_pos % ratio)
    slot_idx = compress_ratio + (start_pos % compress_ratio)

    kv_pcc = compute_pcc(kv_state_cpu[:, slot_idx], kv_state_tt[:, slot_idx])
    score_pcc = compute_pcc(score_state_cpu[:, slot_idx], score_state_tt[:, slot_idx])

    print(f"\nstart_pos={start_pos}, slot_idx={slot_idx}")
    print(f"KV state PCC: {kv_pcc:.6f}, Score state PCC: {score_pcc:.6f}")

    assert kv_pcc >= 0.99, f"KV state PCC too low: {kv_pcc}"
    assert score_pcc >= 0.99, f"Score state PCC too low: {score_pcc}"
    print(f"PASSED: CSA decode accumulate batch_size={batch_size}, start_pos={start_pos}")


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("start_pos", [3, 7])  # Compression positions for ratio=4
def test_ratio_pad_compressor_decode_compress_csa_tt_vs_cpu(batch_size, start_pos):
    """
    Test CSA decode with compression: stub on TT vs original on CPU.

    For ratio=4, compression happens at start_pos=3,7,11,...
    This tests the full decode path including compression output.
    """
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

    # CPU: original - accumulate tokens 0 to start_pos-1 first
    compressor_cpu = setup_compressor(args, compress_ratio, batch_size)

    torch.manual_seed(100)  # Fixed seed for accumulation inputs
    for pos in range(start_pos):
        x = torch.randn(batch_size, 1, args.dim, dtype=torch.bfloat16)
        with torch.no_grad():
            compressor_cpu.forward(x, start_pos=pos)

    # Now the compression token
    x_cpu = torch.randn(batch_size, 1, args.dim, dtype=torch.bfloat16)
    with torch.no_grad():
        out_cpu = compressor_cpu.forward(x_cpu, start_pos=start_pos)

    assert out_cpu is not None, f"Expected output for start_pos={start_pos}, got None"
    print(f"\nCPU original output shape: {out_cpu.shape}")

    # TT: stub - same warmup sequence
    compressor_tt = Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim)
    compressor_tt.load_state_dict(compressor_cpu.state_dict())
    compressor_tt.eval()

    # Reset and replay accumulation on CPU first
    compressor_tt.kv_state.zero_()
    compressor_tt.score_state.fill_(float("-inf"))

    torch.manual_seed(100)  # Same seed for accumulation
    for pos in range(start_pos):
        x = torch.randn(batch_size, 1, args.dim, dtype=torch.bfloat16)
        with torch.no_grad():
            # Use original forward for warmup (on CPU)
            compressor_tt.forward(x, start_pos=pos)

    # Move to TT device after warmup
    compressor_tt = compressor_tt.to(device)
    compressor_tt.kv_cache = torch.zeros(
        batch_size, args.max_seq_len // compress_ratio, args.head_dim,
        dtype=torch.bfloat16, device=device
    )
    compressor_tt.freqs_cis = compressor_cpu.freqs_cis.to(device)

    x_tt = x_cpu.to(device)

    with torch.no_grad():
        out_tt = ratio_pad_compressor_forward(compressor_tt, x_tt, start_pos=start_pos)

    assert out_tt is not None, f"Expected output from TT stub for start_pos={start_pos}"

    xm.mark_step()
    out_tt_cpu = out_tt.cpu()

    print(f"TT stub output shape: {out_tt_cpu.shape}")

    assert out_cpu.shape == out_tt_cpu.shape, \
        f"Shape mismatch: CPU {out_cpu.shape} vs TT {out_tt_cpu.shape}"

    pcc = compute_pcc(out_cpu, out_tt_cpu)
    print(f"PCC: {pcc:.6f}")
    assert pcc >= 0.95, f"PCC too low: {pcc}"
    print(f"PASSED: CSA decode compress batch_size={batch_size}, start_pos={start_pos}")


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("start_pos", [0, 10, 50, 100])  # Accumulate only for ratio=128
def test_ratio_pad_compressor_decode_accumulate_hsa_tt_vs_cpu(batch_size, start_pos):
    """
    Test HSA decode accumulate: stub on TT vs original on CPU.

    For ratio=128, compression happens at start_pos=127,255,...
    """
    xr.set_device_type("TT")
    device = xm.xla_device()
    torch.manual_seed(42)

    args = ModelArgs()
    args.dim = 256
    args.head_dim = 64
    args.rope_head_dim = 16
    args.max_batch_size = batch_size
    args.max_seq_len = 512

    compress_ratio = 128

    # CPU: original (golden)
    compressor_cpu = setup_compressor(args, compress_ratio, batch_size)
    x_cpu = torch.randn(batch_size, 1, args.dim, dtype=torch.bfloat16)

    with torch.no_grad():
        out_cpu = compressor_cpu.forward(x_cpu, start_pos=start_pos)

    assert out_cpu is None, f"Expected None for start_pos={start_pos}"

    kv_state_cpu = compressor_cpu.kv_state[:batch_size].clone()
    score_state_cpu = compressor_cpu.score_state[:batch_size].clone()

    # TT: stub
    compressor_tt = Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim)
    compressor_tt.load_state_dict(compressor_cpu.state_dict())
    compressor_tt.eval()
    compressor_tt.kv_state.zero_()
    compressor_tt.score_state.fill_(float("-inf"))
    compressor_tt = compressor_tt.to(device)

    compressor_tt.kv_cache = torch.zeros(
        batch_size, args.max_seq_len // compress_ratio, args.head_dim,
        dtype=torch.bfloat16, device=device
    )
    compressor_tt.freqs_cis = compressor_cpu.freqs_cis.to(device)

    x_tt = x_cpu.to(device)

    with torch.no_grad():
        out_tt = ratio_pad_compressor_forward(compressor_tt, x_tt, start_pos=start_pos)

    assert out_tt is None

    xm.mark_step()
    kv_state_tt = compressor_tt.kv_state[:batch_size].cpu()
    score_state_tt = compressor_tt.score_state[:batch_size].cpu()

    # For overlap=False, slot_idx = start_pos % ratio
    slot_idx = start_pos % compress_ratio

    kv_pcc = compute_pcc(kv_state_cpu[:, slot_idx], kv_state_tt[:, slot_idx])
    score_pcc = compute_pcc(score_state_cpu[:, slot_idx], score_state_tt[:, slot_idx])

    print(f"\nstart_pos={start_pos}, slot_idx={slot_idx}")
    print(f"KV state PCC: {kv_pcc:.6f}, Score state PCC: {score_pcc:.6f}")

    assert kv_pcc >= 0.99, f"KV state PCC too low: {kv_pcc}"
    assert score_pcc >= 0.99, f"Score state PCC too low: {score_pcc}"
    print(f"PASSED: HSA decode accumulate batch_size={batch_size}, start_pos={start_pos}")


# ============================================================================
# Full Sequence Test - prefill + multiple decode steps
# ============================================================================

@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("prefill_len,decode_steps", [(16, 8), (32, 4)])
def test_ratio_pad_compressor_full_sequence_csa_tt_vs_cpu(batch_size, prefill_len, decode_steps):
    """
    Test full sequence: prefill followed by decode steps.

    CPU uses original Compressor.forward (golden).
    TT uses ratio_pad_compressor_forward stub.
    """
    xr.set_device_type("TT")
    device = xm.xla_device()

    args = ModelArgs()
    args.dim = 256
    args.head_dim = 64
    args.rope_head_dim = 16
    args.max_batch_size = batch_size
    args.max_seq_len = 256

    compress_ratio = 4

    # Generate all inputs upfront with fixed seed
    torch.manual_seed(42)
    x_prefill = torch.randn(batch_size, prefill_len, args.dim, dtype=torch.bfloat16)
    x_decodes = [torch.randn(batch_size, 1, args.dim, dtype=torch.bfloat16)
                 for _ in range(decode_steps)]

    # CPU: original (golden)
    compressor_cpu = setup_compressor(args, compress_ratio, batch_size)

    outputs_cpu = []
    with torch.no_grad():
        out = compressor_cpu.forward(x_prefill, start_pos=0)
        if out is not None:
            outputs_cpu.append(('prefill', out.clone()))

        for i, x_decode in enumerate(x_decodes):
            start_pos = prefill_len + i
            out = compressor_cpu.forward(x_decode, start_pos=start_pos)
            if out is not None:
                outputs_cpu.append((f'decode_{start_pos}', out.clone()))

    print(f"\nCPU original produced {len(outputs_cpu)} outputs")

    # TT: stub
    compressor_tt = Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim)
    compressor_tt.load_state_dict(compressor_cpu.state_dict())
    compressor_tt.eval()
    compressor_tt.kv_state.zero_()
    compressor_tt.score_state.fill_(float("-inf"))
    compressor_tt = compressor_tt.to(device)

    compressor_tt.kv_cache = torch.zeros(
        batch_size, args.max_seq_len // compress_ratio, args.head_dim,
        dtype=torch.bfloat16, device=device
    )
    compressor_tt.freqs_cis = compressor_cpu.freqs_cis.to(device)

    outputs_tt = []
    with torch.no_grad():
        out = ratio_pad_compressor_forward(compressor_tt, x_prefill.to(device), start_pos=0)
        xm.mark_step()
        if out is not None:
            outputs_tt.append(('prefill', out.cpu()))

        for i, x_decode in enumerate(x_decodes):
            start_pos = prefill_len + i
            out = ratio_pad_compressor_forward(compressor_tt, x_decode.to(device), start_pos=start_pos)
            xm.mark_step()
            if out is not None:
                outputs_tt.append((f'decode_{start_pos}', out.cpu()))

    print(f"TT stub produced {len(outputs_tt)} outputs")

    assert len(outputs_cpu) == len(outputs_tt), \
        f"Output count mismatch: CPU {len(outputs_cpu)} vs TT {len(outputs_tt)}"

    for (name_cpu, out_cpu), (name_tt, out_tt) in zip(outputs_cpu, outputs_tt):
        assert name_cpu == name_tt, f"Output order mismatch: {name_cpu} vs {name_tt}"
        assert out_cpu.shape == out_tt.shape, \
            f"Shape mismatch for {name_cpu}: {out_cpu.shape} vs {out_tt.shape}"

        pcc = compute_pcc(out_cpu, out_tt)
        print(f"{name_cpu}: PCC={pcc:.6f}")
        assert pcc >= 0.95, f"PCC too low for {name_cpu}: {pcc}"

    print(f"PASSED: Full sequence prefill_len={prefill_len}, decode_steps={decode_steps}")


# ============================================================================
# Monkey-patch Test - verify install/uninstall works
# ============================================================================

def test_ratio_pad_compressor_monkey_patch():
    """
    Test that install_ratio_pad_compressor properly patches Compressor.forward.
    """
    from kernel_stubs import install_ratio_pad_compressor, uninstall_ratio_pad_compressor

    xr.set_device_type("TT")
    device = xm.xla_device()
    torch.manual_seed(42)

    args = ModelArgs()
    args.dim = 256
    args.head_dim = 64
    args.rope_head_dim = 16
    args.max_batch_size = 1
    args.max_seq_len = 128

    compress_ratio = 4
    batch_size = 1
    seq_len = 32

    # CPU: original (golden)
    compressor_cpu = setup_compressor(args, compress_ratio, batch_size)
    x_cpu = torch.randn(batch_size, seq_len, args.dim, dtype=torch.bfloat16)

    with torch.no_grad():
        out_cpu = compressor_cpu.forward(x_cpu, start_pos=0)

    # TT: monkey-patched forward
    compressor_tt = Compressor(args, compress_ratio=compress_ratio, head_dim=args.head_dim)
    compressor_tt.load_state_dict(compressor_cpu.state_dict())
    compressor_tt.eval()
    compressor_tt = compressor_tt.to(device)

    compressor_tt.kv_cache = torch.zeros(
        batch_size, args.max_seq_len // compress_ratio, args.head_dim,
        dtype=torch.bfloat16, device=device
    )
    compressor_tt.freqs_cis = compressor_cpu.freqs_cis.to(device)

    # Install the patch
    install_ratio_pad_compressor()

    try:
        x_tt = x_cpu.to(device)
        with torch.no_grad():
            # This calls compressor_tt.forward() which is now monkey-patched
            out_tt = compressor_tt.forward(x_tt, start_pos=0)

        xm.mark_step()
        out_tt_cpu = out_tt.cpu()

        pcc = compute_pcc(out_cpu, out_tt_cpu)
        print(f"\nMonkey-patched PCC: {pcc:.6f}")
        assert pcc >= 0.95, f"PCC too low: {pcc}"

    finally:
        # Always uninstall
        uninstall_ratio_pad_compressor()

    print("PASSED: Monkey-patch install/uninstall")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing RatioPadCompressor Stub - CSA Prefill (ratio=4)")
    print("=" * 70)
    test_ratio_pad_compressor_prefill_csa_tt_vs_cpu(batch_size=1, seq_len=32)

    print("\n" + "=" * 70)
    print("Testing RatioPadCompressor Stub - CSA Prefill with Remainder")
    print("=" * 70)
    test_ratio_pad_compressor_prefill_with_remainder_tt_vs_cpu(batch_size=1, seq_len=30)

    print("\n" + "=" * 70)
    print("Testing RatioPadCompressor Stub - HSA Prefill (ratio=128)")
    print("=" * 70)
    test_ratio_pad_compressor_prefill_hsa_tt_vs_cpu(batch_size=1, seq_len=128)

    print("\n" + "=" * 70)
    print("Testing RatioPadCompressor Stub - CSA Decode Accumulate")
    print("=" * 70)
    test_ratio_pad_compressor_decode_accumulate_csa_tt_vs_cpu(batch_size=1, start_pos=1)

    print("\n" + "=" * 70)
    print("Testing RatioPadCompressor Stub - CSA Decode Compress")
    print("=" * 70)
    test_ratio_pad_compressor_decode_compress_csa_tt_vs_cpu(batch_size=1, start_pos=3)

    print("\n" + "=" * 70)
    print("Testing RatioPadCompressor Stub - HSA Decode Accumulate")
    print("=" * 70)
    test_ratio_pad_compressor_decode_accumulate_hsa_tt_vs_cpu(batch_size=1, start_pos=50)

    print("\n" + "=" * 70)
    print("Testing RatioPadCompressor Stub - Full Sequence")
    print("=" * 70)
    test_ratio_pad_compressor_full_sequence_csa_tt_vs_cpu(batch_size=1, prefill_len=16, decode_steps=8)

    print("\n" + "=" * 70)
    print("Testing RatioPadCompressor Stub - Monkey-patch")
    print("=" * 70)
    test_ratio_pad_compressor_monkey_patch()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
