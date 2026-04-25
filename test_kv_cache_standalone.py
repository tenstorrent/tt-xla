#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Standalone test for KV cache update to isolate graph infra issues.
No CPU comparison - just compile and run on TT device.
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


def test_decode_positions():
    """Test decode at different positions without CPU comparison."""
    print("=" * 80)
    print("Standalone KV Cache Update Test - TT Device Only")
    print("=" * 80)

    # Setup
    xr.set_device_type("TT")
    device = xm.xla_device()

    window_size = 8
    head_dim = 32
    bsz = 1
    seqlen = 1  # Decode: single token
    max_batch_size = 2

    # Test positions
    positions = [4, 7]

    # Create SAME input KV for all positions (so we can verify it's written correctly)
    torch.manual_seed(42)
    kv_input = torch.randn(bsz, seqlen, head_dim, dtype=torch.bfloat16).to(device)

    print(f"\n{'=' * 80}")
    print(f"Using SAME input KV for all positions:")
    print(f"  Shape: {kv_input.shape}")
    print(f"  First 8 values: {kv_input[0, 0, :8]}")
    print(f"  Last 8 values: {kv_input[0, 0, -8:]}")
    print(f"{'=' * 80}")

    for start_pos in positions:
        print(f"\n{'─' * 80}")
        print(f"Testing decode at position {start_pos}")
        print(f"{'─' * 80}")

        # Create fresh module for each test
        module = _CircularKVCacheUpdate(window_size, head_dim, max_batch_size).to(
            device
        )

        # Compile with torch.compile for TT backend
        compiled_module = torch.compile(module, backend="tt")

        # Use the SAME input for all tests
        kv = kv_input

        print(
            f"\nExpected: Position {start_pos} should contain the input KV, all other positions should be zero"
        )

        # Run on device
        print("\nCompiling and running on TT device...")
        result = compiled_module(kv, start_pos)

        # Force synchronization
        xm.mark_step()
        xm.wait_device_ops()

        # Get result back to CPU for printing
        result_cpu = result.cpu()
        # print("result for position", start_pos, "is", result.cpu().sum(dim=(0,2)))
        print("result for position", start_pos, "is", result)


if __name__ == "__main__":
    test_decode_positions()
    print("\n" + "=" * 80)
    print("Test complete")
    print("=" * 80)
