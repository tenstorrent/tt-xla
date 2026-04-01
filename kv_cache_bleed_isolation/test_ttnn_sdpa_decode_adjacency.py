#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Standalone TTNN test for paged_scaled_dot_product_attention_decode adjacency bug.

This test reproduces issue #3899 without any XLA/PJRT/vLLM dependency.
It calls the TTNN paged SDPA decode kernel directly and checks whether
adjacent batch items with the same cur_pos produce correct, independent results
when cache blocks contain non-zero padding data.

The bug: when two batch items at adjacent indices have the same cur_pos and
their cache blocks contain non-zero padding (from min_context_len prefill),
the kernel produces incorrect attention output — the padding data leaks through
the causal mask for one of the adjacent items.

Trigger conditions (all required):
  - block_size == 32 (single tile per block)
  - Two batch items at adjacent indices with the same cur_pos
  - Non-zero data at positions cur_pos+1 through block_size-1 in cache blocks

Usage:
  python3 test_ttnn_sdpa_decode_adjacency.py
"""

import torch
import ttnn


def run_paged_sdpa_decode(
    device,
    num_users,
    num_kv_heads,
    num_q_heads,
    head_dim,
    block_size,
    cur_pos_list,
    k_cache_torch,
    v_cache_torch,
    q_torch,
    page_table_torch,
):
    """Run paged SDPA decode on device and return output on CPU."""
    # Move tensors to device
    # Q: [1, num_users, num_q_heads, head_dim]
    q_tt = ttnn.from_torch(
        q_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    # K cache: [num_blocks, num_kv_heads, block_size, head_dim]
    k_tt = ttnn.from_torch(
        k_cache_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    # V cache: [num_blocks, num_kv_heads, block_size, head_dim]
    v_tt = ttnn.from_torch(
        v_cache_torch, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    # Page table: [num_users, max_blocks_per_seq] as int32
    pt_tt = ttnn.from_torch(page_table_torch, device=device, dtype=ttnn.int32)
    # Cur pos: [num_users] as int32
    cur_pos_tensor = torch.tensor(cur_pos_list, dtype=torch.int32)
    cp_tt = ttnn.from_torch(cur_pos_tensor, device=device, dtype=ttnn.int32)

    # Run paged SDPA decode
    out_tt = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_tt,
        k_tt,
        v_tt,
        pt_tt,
        cur_pos_tensor=cp_tt,
        is_causal=True,
    )

    # Move result back to CPU
    out_torch = ttnn.to_torch(out_tt)
    return out_torch


def cpu_reference_paged_sdpa(
    q,
    k_cache,
    v_cache,
    page_table,
    cur_pos_list,
    block_size,
    num_kv_heads,
):
    """CPU reference implementation of paged SDPA decode."""
    num_users = len(cur_pos_list)
    num_q_heads = q.shape[2]
    head_dim = q.shape[3]
    max_blocks = page_table.shape[1]
    max_seq = max_blocks * block_size

    outputs = []
    for u in range(num_users):
        # Gather K/V for this user from page table
        user_k = torch.zeros(num_kv_heads, max_seq, head_dim, dtype=torch.float32)
        user_v = torch.zeros(num_kv_heads, max_seq, head_dim, dtype=torch.float32)
        for b in range(max_blocks):
            phys = page_table[u, b].item()
            start = b * block_size
            user_k[:, start : start + block_size, :] = k_cache[phys].float()
            user_v[:, start : start + block_size, :] = v_cache[phys].float()

        # Causal mask: attend to positions 0..cur_pos
        cp = cur_pos_list[u]
        mask = torch.full((1, max_seq), float("-inf"))
        mask[0, : cp + 1] = 0.0

        # Attention: Q @ K^T * scale + mask -> softmax -> @ V
        scale = 1.0 / (head_dim**0.5)
        user_q = q[0, u].float()  # [num_q_heads, head_dim]

        # GQA: repeat K/V heads to match Q heads
        repeat = num_q_heads // num_kv_heads
        user_k = user_k.repeat_interleave(repeat, dim=0)
        user_v = user_v.repeat_interleave(repeat, dim=0)

        attn = (
            torch.matmul(user_q.unsqueeze(1), user_k.transpose(-1, -2)) * scale
        )  # [heads, 1, seq]
        attn = attn + mask.unsqueeze(0)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, user_v)  # [heads, 1, head_dim]
        outputs.append(out.squeeze(1))

    return (
        torch.stack(outputs, dim=0).unsqueeze(0).bfloat16()
    )  # [1, users, heads, head_dim]


def test_adjacency_bug(device, num_users=8, real_seq_len=14):
    """
    Test that adjacent batch items with the same cur_pos don't interfere.

    Fills cache blocks with distinct per-user real data + uniform padding,
    then checks that swapping which slot a prompt occupies doesn't change output.
    """
    num_kv_heads = 8
    num_q_heads = 8  # Use MHA for simplicity (same heads)
    head_dim = 64
    block_size = 32
    max_blocks_per_seq = 1
    num_blocks = num_users * max_blocks_per_seq

    # Fixed random seed for reproducibility
    torch.manual_seed(42)

    # Create distinct per-user cache data
    k_cache = torch.zeros(
        num_blocks, num_kv_heads, block_size, head_dim, dtype=torch.bfloat16
    )
    v_cache = torch.zeros(
        num_blocks, num_kv_heads, block_size, head_dim, dtype=torch.bfloat16
    )

    for u in range(num_users):
        # Real data at positions 0..real_seq_len-1
        torch.manual_seed(1000 + u)
        k_cache[u, :, :real_seq_len, :] = torch.randn(
            num_kv_heads, real_seq_len, head_dim
        ).bfloat16()
        v_cache[u, :, :real_seq_len, :] = torch.randn(
            num_kv_heads, real_seq_len, head_dim
        ).bfloat16()

        # Non-zero PADDING at positions real_seq_len..block_size-1
        # This simulates what min_context_len=32 produces
        padding_val = torch.randn(num_kv_heads, 1, head_dim).bfloat16() * 2.0
        k_cache[u, :, real_seq_len:, :] = padding_val.expand(
            -1, block_size - real_seq_len, -1
        )
        v_cache[u, :, real_seq_len:, :] = padding_val.expand(
            -1, block_size - real_seq_len, -1
        )

    # Query
    torch.manual_seed(99)
    q = torch.randn(1, num_users, num_q_heads, head_dim, dtype=torch.bfloat16)

    # cur_pos = real_seq_len for all users (simulating first decode step)
    cur_pos_list = [real_seq_len] * num_users

    # === Test 1: Normal order (page_table = identity) ===
    page_table_normal = torch.arange(num_users, dtype=torch.int32).unsqueeze(1)
    # Pad to max_blocks_per_seq columns
    page_table_normal = page_table_normal.expand(-1, max_blocks_per_seq).contiguous()

    out_normal = run_paged_sdpa_decode(
        device,
        num_users,
        num_kv_heads,
        num_q_heads,
        head_dim,
        block_size,
        cur_pos_list,
        k_cache,
        v_cache,
        q,
        page_table_normal,
    )

    # === Test 2: Swap adjacent slots 2 and 3 ===
    # Swap the page_table entries so slot 2 reads from block 3 and vice versa
    page_table_swapped = page_table_normal.clone()
    page_table_swapped[2, 0] = 3
    page_table_swapped[3, 0] = 2

    # Also swap query
    q_swapped = q.clone()
    q_swapped[0, 2] = q[0, 3]
    q_swapped[0, 3] = q[0, 2]

    out_swapped = run_paged_sdpa_decode(
        device,
        num_users,
        num_kv_heads,
        num_q_heads,
        head_dim,
        block_size,
        cur_pos_list,
        k_cache,
        v_cache,
        q_swapped,
        page_table_swapped,
    )

    # After swapping, slot 2's output should match slot 3's original output
    # and slot 3's output should match slot 2's original output
    diff_2to3 = (
        (out_normal[0, 3].float() - out_swapped[0, 2].float()).abs().max().item()
    )
    diff_3to2 = (
        (out_normal[0, 2].float() - out_swapped[0, 3].float()).abs().max().item()
    )

    # Non-swapped slots should be identical
    diff_0 = (out_normal[0, 0].float() - out_swapped[0, 0].float()).abs().max().item()
    diff_1 = (out_normal[0, 1].float() - out_swapped[0, 1].float()).abs().max().item()

    print(f"Adjacent swap test (slots 2↔3):")
    print(f"  Slot 2→3 max_diff: {diff_2to3:.6f}")
    print(f"  Slot 3→2 max_diff: {diff_3to2:.6f}")
    print(f"  Slot 0 unchanged:  {diff_0:.6f}")
    print(f"  Slot 1 unchanged:  {diff_1:.6f}")

    swap_ok = diff_2to3 < 0.1 and diff_3to2 < 0.1
    unchanged_ok = diff_0 < 0.01 and diff_1 < 0.01
    print(f"  Swap consistency: {'PASS' if swap_ok else 'FAIL'}")
    print(f"  Unchanged slots:  {'PASS' if unchanged_ok else 'FAIL'}")

    # === Test 3: CPU reference comparison ===
    cpu_out = cpu_reference_paged_sdpa(
        q,
        k_cache,
        v_cache,
        page_table_normal,
        cur_pos_list,
        block_size,
        num_kv_heads,
    )
    cpu_vs_device = (cpu_out.float() - out_normal.float()).abs()
    per_user_diff = [cpu_vs_device[0, u].max().item() for u in range(num_users)]
    print(f"\nCPU vs Device (normal order):")
    for u in range(num_users):
        print(f"  User {u}: max_diff={per_user_diff[u]:.6f}")
    max_cpu_diff = max(per_user_diff)
    print(f"  Overall max: {max_cpu_diff:.6f}")
    cpu_ok = max_cpu_diff < 1.0  # bf16 tolerance
    print(f"  CPU match: {'PASS' if cpu_ok else 'FAIL'}")

    # === Test 4: Clean padding (zeros) vs dirty padding ===
    k_cache_clean = k_cache.clone()
    v_cache_clean = v_cache.clone()
    for u in range(num_users):
        k_cache_clean[u, :, real_seq_len:, :] = 0
        v_cache_clean[u, :, real_seq_len:, :] = 0

    out_clean = run_paged_sdpa_decode(
        device,
        num_users,
        num_kv_heads,
        num_q_heads,
        head_dim,
        block_size,
        cur_pos_list,
        k_cache_clean,
        v_cache_clean,
        q,
        page_table_normal,
    )
    padding_diff = (out_normal.float() - out_clean.float()).abs()
    max_padding_diff = padding_diff.max().item()
    per_user_padding = [padding_diff[0, u].max().item() for u in range(num_users)]
    print(f"\nPadding leak test (dirty vs clean padding):")
    for u in range(num_users):
        print(f"  User {u}: max_diff={per_user_padding[u]:.6f}")
    print(f"  Overall max: {max_padding_diff:.6f}")
    padding_ok = max_padding_diff < 0.1
    print(f"  Padding masked: {'PASS' if padding_ok else 'FAIL — PADDING LEAKS!'}")

    all_pass = swap_ok and unchanged_ok and cpu_ok and padding_ok
    return all_pass


def main():
    print("=" * 70)
    print("TTNN paged_scaled_dot_product_attention_decode adjacency test")
    print("Issue: https://github.com/tenstorrent/tt-xla/issues/3899")
    print("=" * 70)

    device = ttnn.open_device(device_id=0)
    try:
        # Test with same parameters as the failing vLLM config
        print("\n--- Test: 8 users, seq_len=14, block_size=32 ---")
        result = test_adjacency_bug(device, num_users=8, real_seq_len=14)
        print(f"\nOverall: {'ALL PASS' if result else 'FAIL DETECTED'}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
