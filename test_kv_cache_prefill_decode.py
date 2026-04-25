#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Standalone test for KV cache prefill + decode pattern.
Tests the actual stateful usage: prefill, then decode.
"""

import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


class _CircularKVCacheUpdate(nn.Module):
    """Standalone module that implements circular buffer KV cache update logic."""

    def __init__(self, window_size: int, head_dim: int, max_batch_size: int = 2):
        super().__init__()
        self.win = window_size
        self.register_buffer(
            "kv_cache",
            torch.zeros(max_batch_size, window_size, head_dim, dtype=torch.bfloat16),
            persistent=False,
        )

    def forward(self, kv: torch.Tensor, start_pos: int):
        """
        Update circular buffer with new KV tokens.

        Args:
            kv: New KV tokens [batch, seqlen, head_dim]
            start_pos: Starting position in the sequence

        Returns:
            Updated kv_cache for verification
        """
        bsz, seqlen, _ = kv.shape
        win = self.win

        # Actual logic from Attention.forward (lines 491-502)
        if start_pos == 0:
            # Prefill path
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                # Handle wrap-around for long prefill
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = (
                    kv[:, : win - cutoff],
                    kv[:, win - cutoff :],
                )
        else:
            # Decode path: single token update at circular position
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)

        return self.kv_cache[:bsz]


def test_prefill_4_decode_1():
    """Test: Prefill 4 positions, then decode 1 more (position 4)."""
    print("\n" + "=" * 80)
    print("Test 1: Prefill 4 positions → Decode position 4")
    print("=" * 80)

    xr.set_device_type("TT")
    device = xm.xla_device()

    window_size = 8
    head_dim = 32
    bsz = 1
    max_batch_size = 2

    module = _CircularKVCacheUpdate(window_size, head_dim, max_batch_size).to(device)

    # Compile with torch.compile for TT backend
    compiled_module = torch.compile(module, backend="tt")

    # Prefill: 4 tokens
    torch.manual_seed(100)
    # prefill_kv = torch.randn(bsz, 4, head_dim, dtype=torch.bfloat16).to(device)
    prefill_kv = (
        torch.arange(bsz * 4 * head_dim, dtype=torch.bfloat16)
        .view(bsz, 4, head_dim)
        .to(device)
    )

    print(f"\nPREFILL INPUT: 4 tokens")
    print(prefill_kv)

    cache_after_prefill = compiled_module(prefill_kv, start_pos=0)
    xm.mark_step()
    xm.wait_device_ops()

    print(f"\nCache after PREFILL:")
    print(cache_after_prefill)

    # Decode: 1 token at position 4
    torch.manual_seed(200)
    # decode_kv = torch.randn(bsz, 1, head_dim, dtype=torch.bfloat16).to(device)
    decode_kv = torch.ones(bsz, 1, head_dim, dtype=torch.bfloat16).to(device)
    print(f"\nDECODE INPUT: 1 token at position 4")
    print(decode_kv)

    cache_after_decode = compiled_module(decode_kv, start_pos=4)
    xm.mark_step()
    xm.wait_device_ops()

    print(f"\nCache after DECODE:")
    print(cache_after_decode)


def test_prefill_8_decode_wrap():
    """Test: Prefill 8 positions (full window), then decode 1 more (wraps to position 0)."""
    print("\n" + "=" * 80)
    print("Test 2: Prefill 8 positions (full window) → Decode position 8 (wraps to 0)")
    print("=" * 80)

    xr.set_device_type("TT")
    device = xm.xla_device()

    window_size = 8
    head_dim = 32
    bsz = 1
    max_batch_size = 2

    module = _CircularKVCacheUpdate(window_size, head_dim, max_batch_size).to(device)

    # Compile with torch.compile for TT backend
    compiled_module = torch.compile(module, backend="tt")

    # Prefill: 8 tokens (full window)
    torch.manual_seed(300)
    # prefill_kv = torch.randn(bsz, 8, head_dim, dtype=torch.bfloat16).to(device)
    prefill_kv = (
        torch.arange(bsz * 8 * head_dim, dtype=torch.bfloat16)
        .view(bsz, 8, head_dim)
        .to(device)
    )

    print(f"\nPREFILL INPUT: 8 tokens (fills entire window)")
    print(prefill_kv)

    cache_after_prefill = compiled_module(prefill_kv, start_pos=0)
    xm.mark_step()
    xm.wait_device_ops()

    print(f"\nCache after PREFILL:")
    print(cache_after_prefill)

    # Decode: 1 token at position 8 (should wrap to position 0)
    torch.manual_seed(400)
    # decode_kv = torch.randn(bsz, 1, head_dim, dtype=torch.bfloat16).to(device)
    decode_kv = torch.ones(bsz, 1, head_dim, dtype=torch.bfloat16).to(device)

    print(f"\nDECODE INPUT: 1 token at position 8 (wraps to position 0)")
    print(decode_kv)

    cache_after_decode = compiled_module(decode_kv, start_pos=8)
    xm.mark_step()
    xm.wait_device_ops()

    print(f"\nCache after DECODE:")
    print(cache_after_decode)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("KV Cache Prefill + Decode Pattern Tests")
    print("=" * 80)

    # Test 1: Prefill 4, decode 1
    test_prefill_4_decode_1()

    # Test 2: Prefill 8 (full), decode 1 (wrap)
    test_prefill_8_decode_wrap()

    print("\n" + "=" * 80)
    print("All tests complete")
    print("=" * 80)
