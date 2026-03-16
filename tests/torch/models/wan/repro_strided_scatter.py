# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal repro: combined stride-2 read + write causes tt-mlir compilation hang.

Bug: When a single graph contains both stride-2 reads (x[:, :, :, ::2]) and
stride-2 writes (out[:, :, :, ::2] = val), the tt-mlir compiler hangs during
TTNN binary generation. Each operation in isolation compiles fine (<0.5s).
The combination causes an infinite (or >hours) compilation at seq_len=3120.

This pattern comes from diffusers' apply_rotary_emb in WanAttnProcessor2_0
(diffusers/models/transformers/transformer_wan.py).

Individual ops (all PASS at seq=3120):
    stride-2 read only:   0.2s
    stride-2 write only:  0.5s
    pure math (no stride): 0.5s

Combined (HANGS at seq=3120):
    stride-2 read + math + stride-2 write: >120s (killed)

Workaround: Pre-slice inputs on CPU, use torch.stack+flatten to interleave.

Usage:
    # Quick validation (just the hanging case + workaround):
    python -u tests/torch/models/wan/repro_strided_scatter.py

    # With IR export for inspection:
    TT_TORCH_SAVE_IR=1 python -u tests/torch/models/wan/repro_strided_scatter.py
"""

import gc
import os
import signal
import time

import torch
import torch.nn as nn
import torch_xla
import torch_xla.runtime as xr

xr.set_device_type("TT")
device = torch_xla.device()

SEQ_LEN = 3120  # 480×832 Wan resolution → (2×60/2×104/2) = 3120 patch tokens
NUM_HEADS = 12
HEAD_DIM = 128
HALF_DIM = 64
TIMEOUT_S = 120


def timed_run(model, inputs, name):
    m = model.to(device)
    compiled = torch.compile(m, backend="tt")
    dev_inputs = [x.to(device) for x in inputs]

    def handler(signum, frame):
        raise TimeoutError()

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(TIMEOUT_S)
    try:
        t0 = time.time()
        with torch.no_grad():
            out = compiled(*dev_inputs)
        torch_xla.sync(wait=True)
        elapsed = time.time() - t0
        signal.alarm(0)
        print(f"  {name:<55} {elapsed:>6.1f}s  PASS")
    except TimeoutError:
        signal.alarm(0)
        print(f"  {name:<55} {'>120s':>6}  TIMEOUT  *** BUG ***")
    except Exception as e:
        signal.alarm(0)
        print(f"  {name:<55} {'ERR':>6}  {str(e)[:40]}")
    del compiled, m, dev_inputs
    gc.collect()


# ━━━ BUG: combined stride-2 read + write hangs ━━━━━━━━━━━━━━━━━━━━━━━━━


class BuggyRoPE(nn.Module):
    """Combined stride-2 read + compute + stride-2 write.
    This is the exact pattern from diffusers apply_rotary_emb."""

    def forward(self, qk, freqs):
        x_real = qk[..., 0]  # [B, S, H, half_dim]
        x_imag = qk[..., 1]
        cos_e = freqs[:, :, :, ::2]  # stride-2 read
        sin_e = freqs[:, :, :, 1::2]  # stride-2 read
        c = x_real * cos_e - x_imag * sin_e
        s = x_real * sin_e + x_imag * cos_e
        out = torch.zeros(1, qk.shape[1], NUM_HEADS, HEAD_DIM, device=qk.device)
        out[:, :, :, ::2] = c  # stride-2 write
        out[:, :, :, 1::2] = s  # stride-2 write
        return out


# ━━━ Controls: individual stride ops (all pass) ━━━━━━━━━━━━━━━━━━━━━━━━


class StrideReadOnly(nn.Module):
    def forward(self, freqs):
        return freqs[:, :, :, ::2]


class StrideWriteOnly(nn.Module):
    def forward(self, vals):
        out = torch.zeros(1, SEQ_LEN, NUM_HEADS, HEAD_DIM, device=vals.device)
        out[:, :, :, ::2] = vals
        return out


class PureMathNoStride(nn.Module):
    """Same math as RoPE but with pre-sliced inputs (no stride in graph)."""

    def forward(self, x_real, x_imag, cos_half, sin_half):
        c = x_real * cos_half - x_imag * sin_half
        s = x_real * sin_half + x_imag * cos_half
        return torch.stack([c, s], dim=-1).flatten(-2)


# ━━━ Run ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("=" * 70)
print("tt-mlir compilation hang: combined stride-2 read + write")
print(f"seq_len={SEQ_LEN}, heads={NUM_HEADS}, head_dim={HEAD_DIM}")
print("=" * 70)

# --- Controls (all pass individually, commented out for quick repro) ---
# timed_run(
#     StrideReadOnly(),
#     [torch.randn(1, SEQ_LEN, 1, HEAD_DIM)],
#     "stride-2 read only",
# )
# timed_run(
#     StrideWriteOnly(),
#     [torch.randn(1, SEQ_LEN, NUM_HEADS, HALF_DIM)],
#     "stride-2 write only",
# )
# timed_run(
#     PureMathNoStride(),
#     [
#         torch.randn(1, SEQ_LEN, NUM_HEADS, HALF_DIM),
#         torch.randn(1, SEQ_LEN, NUM_HEADS, HALF_DIM),
#         torch.randn(1, SEQ_LEN, 1, HALF_DIM),
#         torch.randn(1, SEQ_LEN, 1, HALF_DIM),
#     ],
#     "pure math (stack+flatten, no stride)",
# )

print("--- Bug case (should TIMEOUT / HANG) ---")
timed_run(
    BuggyRoPE(),
    [
        torch.randn(1, SEQ_LEN, NUM_HEADS, HALF_DIM, 2),
        torch.randn(1, SEQ_LEN, 1, HEAD_DIM),
    ],
    "combined stride-2 read + write (HANGS)",
)

print("\n" + "=" * 70)
