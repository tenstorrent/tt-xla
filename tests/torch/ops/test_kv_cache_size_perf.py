# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Regression test for KV cache size decode performance.

Reproduces the issue where decode latency scales with total KV cache pool size
(num_blocks) rather than just the active blocks. Increasing
gpu_memory_utilization should not cause proportional decode slowdown when
sequences are short.

In production (vLLM):
- KV cache shape: (num_blocks, num_kv_heads, block_size, head_dim)
  num_blocks scales with gpu_memory_utilization
- Page table shape: (num_users, max_num_blocks_per_seq)
  max_num_blocks_per_seq = max_seq_len / block_size (constant)
- Only a few blocks are active for short sequences, but the full cache
  tensor is passed through the XLA graph every decode step

Observed regression (Llama-3.1-8B, single P150):
  gpu_memory_utilization=0.1 (9.8K tokens):  11.0 tok/s
  gpu_memory_utilization=0.9 (88K tokens):    2.5 tok/s  (4.4x slower)
"""

import time

import pytest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
from infra.utilities.types import Framework

from tests.infra.testers.single_chip.op.op_tester import run_op_test

# Llama-3.2-1B-like: 8 KV heads, head_dim=128, block_size=64
NUM_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 64
NUM_USERS = 1

# max_seq_len=4096 -> 64 blocks per sequence (fixed, independent of cache pool)
MAX_SEQ_LEN = 4096
MAX_BLOCKS_PER_SEQ = MAX_SEQ_LEN // BLOCK_SIZE  # 64

# Active sequence: 4 blocks = 256 tokens of context (short prompt)
ACTIVE_BLOCKS_PER_SEQ = 4
CUR_POS = ACTIVE_BLOCKS_PER_SEQ * BLOCK_SIZE - 1


def _build_decode_inputs(num_blocks, device=None):
    """Build paged attention decode inputs with a fixed short sequence but
    variable total cache pool size (num_blocks)."""
    assert num_blocks >= MAX_BLOCKS_PER_SEQ

    query = torch.randn(
        1, NUM_USERS, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    key = torch.randn(
        num_blocks,
        NUM_KV_HEADS,
        BLOCK_SIZE,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    value = torch.randn(
        num_blocks,
        NUM_KV_HEADS,
        BLOCK_SIZE,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )

    # Page table has fixed width based on max_seq_len, not num_blocks
    page_table = torch.zeros(
        NUM_USERS, MAX_BLOCKS_PER_SEQ, dtype=torch.int32, device=device
    )
    for i in range(ACTIVE_BLOCKS_PER_SEQ):
        page_table[0, i] = i

    cur_pos_tensor = torch.tensor([CUR_POS], dtype=torch.int32, device=device)
    update_indices = torch.tensor([CUR_POS], dtype=torch.int32, device=device)

    return query, key, value, page_table, cur_pos_tensor, update_indices


def _bench_fn(fn, warmup_iters=3, bench_iters=10):
    """Run a function with warmup and return avg time in ms."""
    for _ in range(warmup_iters):
        fn()
    times = []
    for _ in range(bench_iters):
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return sum(times) / len(times) * 1000


# --- Correctness: verify decode results are independent of cache pool size ---


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("num_blocks", [64, 256, 1024])
def test_paged_attention_decode_correctness_vs_cache_size(num_blocks):
    """Verify decode produces correct results regardless of total cache size."""
    q, k, v, pt, cur_pos, _ = _build_decode_inputs(num_blocks)
    run_op_test(
        torch.ops.tt.paged_scaled_dot_product_attention_decode,
        [q, k, v, pt, True, None, cur_pos],
        framework=Framework.TORCH,
    )


# --- Perf: paged attention kernel in isolation ---


@pytest.mark.single_device
@pytest.mark.parametrize(
    "small_blocks,large_blocks",
    [
        (64, 1024),  # 16x cache size increase
        (64, 2048),  # 32x cache size increase
    ],
    ids=["64v1024", "64v2048"],
)
def test_kv_cache_size_decode_perf_attention_only(small_blocks, large_blocks):
    """Paged attention decode: latency should not scale with total cache pool.

    Tests the TTNN paged attention kernel in isolation. Page table has fixed
    width (max_seq_len / block_size) — only the KV cache tensor size varies.

    NOTE: The production regression (4.4x at 9x cache size) may not reproduce
    at the single-op level. The full model graph passes 32 layers × 2 cache
    tensors per decode step, and the aggregate data movement is what likely
    causes the regression. This test establishes the kernel-level baseline.
    """
    device = torch_xla.device()

    configs = [("small", small_blocks), ("large", large_blocks)]
    results = {}

    for label, num_blocks in configs:
        q, k, v, pt, cur_pos, _ = _build_decode_inputs(num_blocks, device=device)

        def step():
            torch.ops.tt.paged_scaled_dot_product_attention_decode(
                q, k, v, pt, True, None, cur_pos
            )
            xm.mark_step()
            xm.wait_device_ops()

        avg_ms = _bench_fn(step)
        results[label] = avg_ms
        cache_mb = num_blocks * NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM * 2 / (1024 * 1024)
        print(
            f"  [attn] num_blocks={num_blocks:>5} ({label:>5}): "
            f"{avg_ms:.2f} ms ({cache_mb:.0f} MB per K or V)"
        )

    ratio = results["large"] / results["small"]
    print(f"  [attn] Ratio: {ratio:.2f}x")

    assert ratio < 2.0, (
        f"Attention-only decode regression: {ratio:.1f}x slower. "
        f"small={results['small']:.2f}ms, large={results['large']:.2f}ms"
    )


# --- Perf: cache update + attention (one layer's decode step) ---


@pytest.mark.single_device
@pytest.mark.xfail(
    reason="paged_update_cache TTIR->TTNN lowering fails (Error code: 13)",
    strict=False,
)
@pytest.mark.parametrize(
    "small_blocks,large_blocks",
    [(64, 1024)],
    ids=["64v1024"],
)
def test_kv_cache_size_decode_perf_update_and_attention(small_blocks, large_blocks):
    """Cache update + attention: one transformer layer's full decode step.

    paged_update_cache returns a new cache tensor (functional XLA semantics),
    then paged_attention reads from the updated cache. Tests whether the full
    copy semantics of cache update cause regression with larger pools.

    Currently xfail: paged_update_cache doesn't compile on this build.
    """
    device = torch_xla.device()

    configs = [("small", small_blocks), ("large", large_blocks)]
    results = {}

    for label, num_blocks in configs:
        q, k, v, pt, cur_pos, update_idx = _build_decode_inputs(
            num_blocks, device=device
        )
        new_kv = torch.randn(
            1,
            NUM_USERS,
            NUM_KV_HEADS,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )

        def step():
            k_updated = torch.ops.tt.paged_update_cache(k, new_kv, update_idx, pt)
            v_updated = torch.ops.tt.paged_update_cache(v, new_kv, update_idx, pt)
            torch.ops.tt.paged_scaled_dot_product_attention_decode(
                q, k_updated, v_updated, pt, True, None, cur_pos
            )
            xm.mark_step()
            xm.wait_device_ops()

        avg_ms = _bench_fn(step)
        results[label] = avg_ms
        cache_mb = (
            2 * num_blocks * NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM * 2 / (1024 * 1024)
        )
        print(
            f"  [update+attn] num_blocks={num_blocks:>5} ({label:>5}): "
            f"{avg_ms:.2f} ms ({cache_mb:.0f} MB K+V)"
        )

    ratio = results["large"] / results["small"]
    print(f"  [update+attn] Ratio: {ratio:.2f}x")

    assert ratio < 2.0, (
        f"Update+attention decode regression: {ratio:.1f}x slower. "
        f"small={results['small']:.2f}ms, large={results['large']:.2f}ms"
    )
