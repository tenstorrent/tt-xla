#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test whether paged_scaled_dot_product_attention_decode leaks padding data.

Hypothesis: when min_context_len pads a 14-token prompt to 32 tokens,
positions 14-31 in the cache block contain garbage. If the decode attention
kernel reads all 32 positions instead of masking beyond cache_position,
the garbage corrupts the output.

This test:
1. Fills a cache block with real data at positions 0-13 and LARGE garbage at 14-31
2. Runs paged_scaled_dot_product_attention_decode with cur_pos_tensor=14
3. Compares output to a clean cache (zeros at 14-31)
4. If outputs differ, the kernel is leaking padding data

Run: python3 kv_cache_bleed_isolation/test_attention_padding_leak.py
"""

import torch
import tt_torch.custom_ops  # noqa: F401


def test_padding_leak_cpu():
    """Test on CPU first to establish baseline behavior."""
    num_users = 4
    num_heads = 8  # GQA heads for Llama-3.2-1B
    head_dim = 64
    block_size = 32
    num_blocks = num_users * 2  # 2 blocks per user
    max_blocks_per_seq = 2
    real_seq_len = 14  # submarine's prompt length

    # Non-overlapping page tables
    page_table = torch.zeros(num_users, max_blocks_per_seq, dtype=torch.int32)
    for u in range(num_users):
        for b in range(max_blocks_per_seq):
            page_table[u, b] = u * max_blocks_per_seq + b

    cache_position = torch.full((num_users,), real_seq_len, dtype=torch.int32)

    # Query: distinct per user
    query = torch.randn(1, num_users, num_heads * 4, head_dim, dtype=torch.bfloat16)
    # Llama GQA: 32 query heads, 8 KV heads → query has 32 heads but cache has 8

    # Actually, for the custom op, query should match the attention head count
    # Let's use num_heads for simplicity (the kernel handles GQA internally)
    query = torch.randn(1, num_users, num_heads, head_dim, dtype=torch.bfloat16)

    # === Clean cache: zeros beyond real_seq_len ===
    k_cache_clean = torch.zeros(
        num_blocks, num_heads, block_size, head_dim, dtype=torch.bfloat16
    )
    v_cache_clean = torch.zeros(
        num_blocks, num_heads, block_size, head_dim, dtype=torch.bfloat16
    )

    # Fill real positions with distinct per-user data
    for u in range(num_users):
        phys_block = page_table[u, 0].item()
        for pos in range(real_seq_len):
            fill_val = (u + 1) * 0.01 * (pos + 1)
            k_cache_clean[phys_block, :, pos, :] = fill_val
            v_cache_clean[phys_block, :, pos, :] = fill_val

    # === Dirty cache: garbage beyond real_seq_len ===
    k_cache_dirty = k_cache_clean.clone()
    v_cache_dirty = v_cache_clean.clone()

    # Fill padding positions with LARGE garbage (simulates what min_context_len does)
    for u in range(num_users):
        phys_block = page_table[u, 0].item()
        for pos in range(real_seq_len, block_size):
            # Use a distinctive large value that would clearly affect softmax
            garbage_val = 10.0 * (u + 1)
            k_cache_dirty[phys_block, :, pos, :] = garbage_val
            v_cache_dirty[phys_block, :, pos, :] = garbage_val

    # Run attention on CPU with clean cache
    out_clean = torch.ops.tt.paged_scaled_dot_product_attention_decode(
        query,
        k_cache_clean,
        v_cache_clean,
        page_table,
        cur_pos_tensor=cache_position,
        is_causal=True,
    )

    # Run attention on CPU with dirty cache
    out_dirty = torch.ops.tt.paged_scaled_dot_product_attention_decode(
        query,
        k_cache_dirty,
        v_cache_dirty,
        page_table,
        cur_pos_tensor=cache_position,
        is_causal=True,
    )

    # Compare
    diff = (out_clean.float() - out_dirty.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"CPU test: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
    if max_diff > 0.01:
        print("  CPU LEAKS padding data!")
        for u in range(num_users):
            u_diff = diff[0, u].max().item()
            print(f"  User {u}: max_diff={u_diff:.6f}")
    else:
        print("  CPU correctly masks padding")
    return max_diff


def test_padding_leak_device():
    """Test on TT device to check TTNN kernel behavior."""
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    print(f"\nDevice: {device}")

    num_users = 4
    num_heads = 8
    head_dim = 64
    block_size = 32
    num_blocks = num_users * 2
    max_blocks_per_seq = 2
    real_seq_len = 14

    page_table = torch.zeros(num_users, max_blocks_per_seq, dtype=torch.int32)
    for u in range(num_users):
        for b in range(max_blocks_per_seq):
            page_table[u, b] = u * max_blocks_per_seq + b

    cache_position = torch.full((num_users,), real_seq_len, dtype=torch.int32)
    query = torch.randn(1, num_users, num_heads, head_dim, dtype=torch.bfloat16)

    # Clean cache
    k_clean = torch.zeros(
        num_blocks, num_heads, block_size, head_dim, dtype=torch.bfloat16
    )
    v_clean = torch.zeros(
        num_blocks, num_heads, block_size, head_dim, dtype=torch.bfloat16
    )
    for u in range(num_users):
        phys = page_table[u, 0].item()
        for pos in range(real_seq_len):
            k_clean[phys, :, pos, :] = (u + 1) * 0.01 * (pos + 1)
            v_clean[phys, :, pos, :] = (u + 1) * 0.01 * (pos + 1)

    # Dirty cache
    k_dirty = k_clean.clone()
    v_dirty = v_clean.clone()
    for u in range(num_users):
        phys = page_table[u, 0].item()
        for pos in range(real_seq_len, block_size):
            k_dirty[phys, :, pos, :] = 10.0 * (u + 1)
            v_dirty[phys, :, pos, :] = 10.0 * (u + 1)

    # Move to device
    query_d = query.to(device)
    pt_d = page_table.to(device)
    cp_d = cache_position.to(device)
    k_clean_d = k_clean.to(device)
    v_clean_d = v_clean.to(device)
    k_dirty_d = k_dirty.to(device)
    v_dirty_d = v_dirty.to(device)

    # Run clean
    out_clean_d = torch.ops.tt.paged_scaled_dot_product_attention_decode(
        query_d,
        k_clean_d,
        v_clean_d,
        pt_d,
        cur_pos_tensor=cp_d,
        is_causal=True,
    )
    out_clean = out_clean_d.cpu()

    # Run dirty
    out_dirty_d = torch.ops.tt.paged_scaled_dot_product_attention_decode(
        query_d,
        k_dirty_d,
        v_dirty_d,
        pt_d,
        cur_pos_tensor=cp_d,
        is_causal=True,
    )
    out_dirty = out_dirty_d.cpu()

    diff = (out_clean.float() - out_dirty.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Device test: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
    if max_diff > 0.01:
        print("  DEVICE LEAKS padding data! This is the bug!")
        for u in range(num_users):
            u_diff = diff[0, u].max().item()
            print(f"  User {u}: max_diff={u_diff:.6f}")
    else:
        print("  Device correctly masks padding")

    # Also compare device clean vs CPU clean
    cpu_out = torch.ops.tt.paged_scaled_dot_product_attention_decode(
        query,
        k_clean,
        v_clean,
        page_table,
        cur_pos_tensor=cache_position,
        is_causal=True,
    )
    cpu_vs_dev = (cpu_out.float() - out_clean.float()).abs().max().item()
    print(f"  CPU vs Device (clean cache): max_diff={cpu_vs_dev:.6f}")

    return max_diff


if __name__ == "__main__":
    print("=" * 60)
    print("Testing paged attention decode padding leak")
    print("=" * 60)

    cpu_diff = test_padding_leak_cpu()

    try:
        dev_diff = test_padding_leak_device()
    except Exception as e:
        print(f"\nDevice test failed: {e}")
        dev_diff = -1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"CPU padding leak:    {'YES' if cpu_diff > 0.01 else 'NO'} (max_diff={cpu_diff:.6f})"
    )
    if dev_diff >= 0:
        print(
            f"Device padding leak: {'YES' if dev_diff > 0.01 else 'NO'} (max_diff={dev_diff:.6f})"
        )
