# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test paged_update_cache → paged_scaled_dot_product_attention_decode roundtrip
through torch.compile(backend="tt") to isolate whether bleed occurs in the
compiled kernel pipeline vs. vLLM orchestration.

This bypasses the full model — just tests cache write + attention read
with controlled inputs and multiple batch slots.
"""

import torch
import torch.nn as nn

# Import TT custom ops
import tt_torch.custom_ops  # noqa: F401


class CacheUpdateAndAttend(nn.Module):
    """Minimal module: update KV cache then run paged attention decode."""

    def forward(
        self,
        query: torch.Tensor,  # [1, num_users, num_heads, head_dim]
        key: torch.Tensor,  # [1, num_users, num_heads, head_dim]
        value: torch.Tensor,  # [1, num_users, num_heads, head_dim]
        k_cache: torch.Tensor,  # [num_blocks, num_heads, block_size, head_dim]
        v_cache: torch.Tensor,  # [num_blocks, num_heads, block_size, head_dim]
        cache_position: torch.Tensor,  # [num_users]
        page_table: torch.Tensor,  # [num_users, max_blocks_per_seq]
    ):
        # Update cache (same as attention.py decode path)
        k_cache = torch.ops.tt.paged_update_cache(
            k_cache, key, cache_position, page_table
        )
        v_cache = torch.ops.tt.paged_update_cache(
            v_cache, value, cache_position, page_table
        )

        # Attention decode (reads from updated cache)
        out = torch.ops.tt.paged_scaled_dot_product_attention_decode(
            query,
            k_cache,
            v_cache,
            page_table,
            cur_pos_tensor=cache_position,
            is_causal=True,
        )
        return out, k_cache, v_cache


def run_cpu_reference(query, key, value, k_cache, v_cache, cache_position, page_table):
    """Run the same operations on CPU for comparison."""
    model = CacheUpdateAndAttend()
    with torch.no_grad():
        return model(query, key, value, k_cache, v_cache, cache_position, page_table)


def create_test_inputs(
    num_users=4,
    num_heads=12,
    head_dim=64,
    block_size=32,
    num_blocks=16,
    max_blocks_per_seq=4,
    seq_position=5,
    device="cpu",
):
    """Create controlled test inputs with distinct per-slot values."""
    # Each user gets a unique fill value so we can detect bleed
    key = torch.zeros(1, num_users, num_heads, head_dim, dtype=torch.bfloat16)
    value = torch.zeros(1, num_users, num_heads, head_dim, dtype=torch.bfloat16)
    query = torch.zeros(1, num_users, num_heads, head_dim, dtype=torch.bfloat16)

    for u in range(num_users):
        fill_val = (u + 1) * 0.1  # 0.1, 0.2, 0.3, 0.4
        key[0, u, :, :] = fill_val
        value[0, u, :, :] = fill_val
        query[0, u, :, :] = fill_val

    # Initialize cache with zeros
    k_cache = torch.zeros(
        num_blocks, num_heads, block_size, head_dim, dtype=torch.bfloat16
    )
    v_cache = torch.zeros(
        num_blocks, num_heads, block_size, head_dim, dtype=torch.bfloat16
    )

    # Pre-fill cache with user-specific data at earlier positions
    for u in range(num_users):
        for pos in range(seq_position):
            block_idx = pos // block_size
            block_offset = pos % block_size
            phys_block = u * max_blocks_per_seq + block_idx
            fill = (u + 1) * 0.01 * (pos + 1)
            k_cache[phys_block, :, block_offset, :] = fill
            v_cache[phys_block, :, block_offset, :] = fill

    # Non-overlapping page tables
    page_table = torch.zeros(num_users, max_blocks_per_seq, dtype=torch.int32)
    for u in range(num_users):
        for b in range(max_blocks_per_seq):
            page_table[u, b] = u * max_blocks_per_seq + b

    cache_position = torch.full((num_users,), seq_position, dtype=torch.int32)

    if device != "cpu":
        key = key.to(device)
        value = value.to(device)
        query = query.to(device)
        k_cache = k_cache.to(device)
        v_cache = v_cache.to(device)
        page_table = page_table.to(device)
        cache_position = cache_position.to(device)

    return query, key, value, k_cache, v_cache, cache_position, page_table


def check_cache_isolation(
    k_cache_out, v_cache_out, page_table, num_users, max_blocks_per_seq
):
    """Check that each user's cache blocks only contain that user's data."""
    violations = []
    for u in range(num_users):
        expected_val_approx = (u + 1) * 0.1
        for b in range(max_blocks_per_seq):
            phys_block = page_table[u, b].item()
            # Check the last written position in the block (position 5 → block 0, offset 5)
            block_data = k_cache_out[phys_block, 0, 5, 0].item()
            if block_data != 0.0:
                # Check if value belongs to the correct user
                for other_u in range(num_users):
                    other_val = (other_u + 1) * 0.1
                    if other_u != u and abs(block_data - other_val) < 0.05:
                        violations.append(
                            f"User {u} block {phys_block} has value {block_data:.4f} "
                            f"(expected ~{expected_val_approx:.2f}, looks like user {other_u})"
                        )
    return violations


def compare_attention_outputs(cpu_out, device_out, num_users):
    """Compare per-user attention outputs between CPU and device."""
    mismatches = []
    for u in range(num_users):
        cpu_val = cpu_out[0, u, 0, 0].item()
        dev_val = device_out[0, u, 0, 0].item()
        if abs(cpu_val - dev_val) > 0.1:
            mismatches.append(
                f"User {u}: cpu={cpu_val:.6f}, device={dev_val:.6f}, "
                f"diff={abs(cpu_val - dev_val):.6f}"
            )

    # Cross-user check: verify each user's output is distinct
    for u1 in range(num_users):
        for u2 in range(u1 + 1, num_users):
            val1 = device_out[0, u1, 0, 0].item()
            val2 = device_out[0, u2, 0, 0].item()
            if abs(val1 - val2) < 1e-6:
                mismatches.append(
                    f"Users {u1} and {u2} have IDENTICAL outputs: {val1:.6f} — possible bleed"
                )

    return mismatches


def test_compiled_roundtrip(num_users=4, num_iterations=10):
    """Main test: compile cache+attention and verify isolation."""
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    print(f"Device: {device}")
    print(f"Testing with {num_users} users, {num_iterations} iterations")

    model = CacheUpdateAndAttend()
    compiled_model = torch.compile(model, backend="tt")

    num_heads = 12
    head_dim = 64
    block_size = 32
    num_blocks = num_users * 4  # 4 blocks per user
    max_blocks_per_seq = 4

    total_failures = 0

    for iteration in range(num_iterations):
        seq_pos = 5 + iteration  # Advance position each iteration

        # CPU reference
        cpu_inputs = create_test_inputs(
            num_users=num_users,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
            num_blocks=num_blocks,
            max_blocks_per_seq=max_blocks_per_seq,
            seq_position=seq_pos,
            device="cpu",
        )
        cpu_out, cpu_k, cpu_v = run_cpu_reference(*cpu_inputs)

        # Device
        dev_inputs = create_test_inputs(
            num_users=num_users,
            num_heads=num_heads,
            head_dim=head_dim,
            block_size=block_size,
            num_blocks=num_blocks,
            max_blocks_per_seq=max_blocks_per_seq,
            seq_position=seq_pos,
            device=device,
        )

        with torch.no_grad():
            dev_out, dev_k, dev_v = compiled_model(*dev_inputs)

        # Move results to CPU
        dev_out_cpu = dev_out.cpu()
        dev_k_cpu = dev_k.cpu()
        dev_v_cpu = dev_v.cpu()

        # Check cache isolation
        cache_violations = check_cache_isolation(
            dev_k_cpu,
            dev_v_cpu,
            cpu_inputs[6],  # page_table
            num_users,
            max_blocks_per_seq,
        )

        # Compare attention outputs
        attn_mismatches = compare_attention_outputs(cpu_out, dev_out_cpu, num_users)

        failed = bool(cache_violations or attn_mismatches)
        if failed:
            total_failures += 1
            print(f"\nIteration {iteration}: FAIL")
            for v in cache_violations:
                print(f"  CACHE: {v}")
            for m in attn_mismatches:
                print(f"  ATTN:  {m}")
        else:
            print(f"Iteration {iteration}: PASS", end="  ")

    print(f"\n\nResults: {total_failures}/{num_iterations} failures")
    return total_failures


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-users", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    failures = test_compiled_roundtrip(
        num_users=args.num_users, num_iterations=args.iterations
    )
    exit(1 if failures > 0 else 0)
