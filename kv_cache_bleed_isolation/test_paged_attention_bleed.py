#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# Non-vLLM KV cache bleed test.
#
# Directly exercises the paged_scaled_dot_product_attention_decode op
# with controlled block tables and known cache contents. No vLLM needed.
#
# Each batch slot gets:
#   - Distinct KV cache content (slot_id * 0.1 fill value)
#   - Non-overlapping physical blocks in the page table
#   - Same query (to make output comparison easy)
#
# If the attention op correctly isolates slots, each slot's output should
# only reflect its own cache content. Any cross-slot influence = bleed.
#
# Usage:
#   python3 test_paged_attention_bleed.py
#   pytest test_paged_attention_bleed.py -v

import os
import sys

import torch
import torch_xla

# Must import tt_torch to register custom ops
import tt_torch  # noqa: F401


def create_test_inputs(
    num_users=16,
    num_heads=8,
    head_dim=128,
    block_size=64,
    blocks_per_seq=4,
    seq_len=200,
):
    """Create controlled test inputs for paged attention decode.

    Each user slot gets:
    - Unique KV cache values (scaled by slot index)
    - Non-overlapping physical blocks
    - Identical query vectors
    """
    total_blocks = blocks_per_seq * num_users

    # KV caches: fill each user's blocks with a distinct pattern
    k_cache = torch.zeros(
        total_blocks, num_heads, block_size, head_dim, dtype=torch.bfloat16
    )
    v_cache = torch.zeros(
        total_blocks, num_heads, block_size, head_dim, dtype=torch.bfloat16
    )

    # Page table: user i gets physical blocks [i*blocks_per_seq, ..., (i+1)*blocks_per_seq-1]
    page_table = torch.zeros(num_users, blocks_per_seq, dtype=torch.int32)

    for user in range(num_users):
        # Assign non-overlapping physical blocks
        for b in range(blocks_per_seq):
            phys_block = user * blocks_per_seq + b
            page_table[user, b] = phys_block

            # Fill with user-specific pattern: user 0 gets 0.1, user 1 gets 0.2, etc.
            fill_val = (user + 1) * 0.1
            k_cache[phys_block, :, :, :] = fill_val
            v_cache[phys_block, :, :, :] = fill_val

    # Query: same for all users (so output differences must come from cache)
    query = torch.ones(1, num_users, num_heads, head_dim, dtype=torch.bfloat16)

    # Cache position: all users at the same sequence position
    cache_position = torch.full(
        (num_users,), min(seq_len, blocks_per_seq * block_size) - 1, dtype=torch.int32
    )

    return query, k_cache, v_cache, page_table, cache_position


def check_output_isolation(outputs, num_users, tolerance=0.05):
    """Check that each user's output is distinct and consistent.

    If slot i's output looks like slot j's output, there's bleed.
    Returns list of (source, victim) bleed pairs.
    """
    bleeds = []

    # Compute per-user output fingerprint (mean value)
    fingerprints = []
    for user in range(num_users):
        fp = outputs[0, user, :, :].float().mean().item()
        fingerprints.append(fp)

    # Check each pair — outputs should be proportional to cache fill values
    for i in range(num_users):
        for j in range(i + 1, num_users):
            # If two users have very similar output despite different cache content
            if abs(fingerprints[i] - fingerprints[j]) < tolerance:
                # They shouldn't be similar unless one is contaminating the other
                bleeds.append((i, j, fingerprints[i], fingerprints[j]))

    return fingerprints, bleeds


def run_paged_attention_on_device(query, k_cache, v_cache, page_table, cache_position):
    """Run paged_scaled_dot_product_attention_decode on TT device."""

    @torch.compile(backend="tt")
    def attention_fn(q, k, v, pt, cp):
        return torch.ops.tt.paged_scaled_dot_product_attention_decode(
            q, k, v, pt, True, None, cp
        )

    output = attention_fn(query, k_cache, v_cache, page_table, cache_position)
    return output


def run_test(num_users=16, num_runs=20):
    """Run the bleed test multiple times."""
    print(f"Users/batch: {num_users}")
    print(f"Runs:        {num_runs}")
    print()

    query, k_cache, v_cache, page_table, cache_position = create_test_inputs(
        num_users=num_users
    )

    passes = 0
    fails = 0

    for run in range(1, num_runs + 1):
        output = run_paged_attention_on_device(
            query, k_cache, v_cache, page_table, cache_position
        )

        fingerprints, bleeds = check_output_isolation(output, num_users)

        if bleeds:
            fails += 1
            for src, vic, fp_src, fp_vic in bleeds:
                print(
                    f"  BLEED: slot {src} (fp={fp_src:.4f}) ≈ slot {vic} (fp={fp_vic:.4f})"
                )
            print(f"  Fingerprints: {[f'{f:.4f}' for f in fingerprints]}")
            print(f"  Run {run}/{num_runs}: FAIL\n")
        else:
            passes += 1
            print(f"  Run {run}/{num_runs}: PASS  [{passes}P/{fails}F]")

    print()
    print("============================================")
    print(f"Results: {passes} PASS / {fails} FAIL out of {num_runs}")
    if fails > 0:
        print(f"Failure rate: {fails / num_runs * 100:.1f}%")
    print("============================================")
    return fails


# ---- Pytest test ----


def test_paged_attention_no_bleed():
    """Non-vLLM paged attention bleed test.

    Directly exercises the paged_scaled_dot_product_attention_decode op
    with controlled block tables. Each slot gets distinct cache content
    and non-overlapping physical blocks. Bleed = output contamination.
    """
    num_users = 16
    num_runs = 10

    query, k_cache, v_cache, page_table, cache_position = create_test_inputs(
        num_users=num_users
    )

    all_bleeds = []
    for run in range(num_runs):
        output = run_paged_attention_on_device(
            query, k_cache, v_cache, page_table, cache_position
        )
        fingerprints, bleeds = check_output_isolation(output, num_users)
        for src, vic, fp_src, fp_vic in bleeds:
            all_bleeds.append(
                f"Run {run+1}: slot {src} (fp={fp_src:.4f}) ≈ slot {vic} (fp={fp_vic:.4f})"
            )

    assert (
        len(all_bleeds) == 0
    ), f"Paged attention output bleed detected:\n" + "\n".join(all_bleeds[:10])


if __name__ == "__main__":
    num_users = int(os.environ.get("NUM_USERS", 16))
    num_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    fails = run_test(num_users=num_users, num_runs=num_runs)
    sys.exit(1 if fails > 0 else 0)
