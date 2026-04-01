#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# Compare paged_update_cache behavior between CPU and TT device.
#
# Fills a KV cache with distinct per-slot patterns using paged_update_cache,
# then compares the resulting cache between CPU and device execution.
# Any divergence indicates a bug in the device kernel or compiled graph.
#
# Usage:
#   python3 test_cache_write_compare.py

import os
import sys

import torch
import torch_xla
import tt_torch  # noqa: F401 — registers custom ops


def test_paged_update_cache_isolation(
    num_users=16,
    num_heads=8,
    head_dim=128,
    block_size=64,
    blocks_per_seq=2,
):
    """Write distinct values per slot, verify no cross-slot contamination."""

    total_blocks = blocks_per_seq * num_users

    # Page table: user i gets blocks [i*blocks_per_seq, ..., (i+1)*blocks_per_seq-1]
    page_table = torch.zeros(num_users, blocks_per_seq, dtype=torch.int32)
    for user in range(num_users):
        for b in range(blocks_per_seq):
            page_table[user, b] = user * blocks_per_seq + b

    # Simulate multiple decode steps, each writing one token per user
    num_steps = 10
    results = {"cpu": [], "device": []}

    for mode in ["cpu", "device"]:
        cache = torch.zeros(
            total_blocks, num_heads, block_size, head_dim, dtype=torch.bfloat16
        )

        for step in range(num_steps):
            # Each user writes a unique value: user_id * 0.1 + step * 0.001
            fill_value = torch.zeros(
                1, num_users, num_heads, head_dim, dtype=torch.bfloat16
            )
            for user in range(num_users):
                fill_value[0, user, :, :] = (user + 1) * 0.1 + step * 0.001

            # update_indices: position in the sequence (step index)
            update_indices = torch.full((num_users,), step, dtype=torch.int32)

            if mode == "cpu":
                cache = torch.ops.tt.paged_update_cache(
                    cache, fill_value, update_indices, page_table
                )
            else:

                @torch.compile(backend="tt")
                def update_cache_tt(c, fv, ui, pt):
                    return torch.ops.tt.paged_update_cache(c, fv, ui, pt)

                cache = update_cache_tt(cache, fill_value, update_indices, page_table)

        results[mode] = cache

    # Compare CPU vs device results
    cpu_cache = results["cpu"]
    dev_cache = results["device"]

    if dev_cache.device.type != "cpu":
        dev_cache = dev_cache.cpu()

    print(
        f"Config: {num_users} users, {num_heads} heads, block_size={block_size}, {num_steps} steps"
    )
    print()

    # Check per-slot isolation
    bleed_found = False
    for user in range(num_users):
        for b in range(blocks_per_seq):
            phys_block = page_table[user, b].item()
            cpu_block = cpu_cache[phys_block]
            dev_block = dev_cache[phys_block]

            # Check if device result matches CPU
            max_diff = (cpu_block.float() - dev_block.float()).abs().max().item()

            if max_diff > 0.01:
                print(
                    f"  MISMATCH: Slot {user} block {phys_block}: max_diff={max_diff:.6f}"
                )
                # Check which other slot's values appear in this block
                for other_user in range(num_users):
                    if other_user == user:
                        continue
                    expected_val = (other_user + 1) * 0.1
                    if (dev_block.float() - expected_val).abs().min().item() < 0.01:
                        print(
                            f"    Contains values from slot {other_user} (expected ~{expected_val:.1f})"
                        )
                        bleed_found = True

    # Also check for cross-slot contamination directly
    print("\nPer-slot value check (device):")
    for user in range(num_users):
        phys_block = page_table[user, 0].item()
        # Check values written in the first few steps
        for step in range(min(3, num_steps)):
            dev_val = dev_cache[phys_block, 0, step, 0].item()  # First head, first dim
            expected = (user + 1) * 0.1 + step * 0.001
            match = (
                "OK" if abs(dev_val - expected) < 0.01 else f"WRONG (got {dev_val:.4f})"
            )
            if user < 6 or "WRONG" in match:  # Only print first 6 + any errors
                print(
                    f"  Slot {user:2d} step {step}: expected={expected:.4f} got={dev_val:.4f} {match}"
                )

    # Overall comparison
    total_diff = (cpu_cache.float() - dev_cache.float()).abs().max().item()
    print(f"\nMax CPU vs device divergence: {total_diff:.6f}")

    if bleed_found:
        print("\nBLEED DETECTED in cache write path!")
        return False
    elif total_diff > 0.01:
        print(f"\nSIGNIFICANT DIVERGENCE between CPU and device (max={total_diff:.6f})")
        return False
    else:
        print("\nPASS — cache writes match between CPU and device")
        return True


if __name__ == "__main__":
    passed = test_paged_update_cache_isolation()
    sys.exit(0 if passed else 1)
