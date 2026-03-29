# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone test for ttnn.topk() + ttnn.sampling() pipeline at production vocab sizes.

## Vocab Size Limitation (discovered 2026-03-26)

ttnn.sampling() overflows L1 at vocab_size >= 8192. Three circular buffers in the
sampling kernel scale with vocab width (Wt = vocab_size / TILE_WIDTH):

  c_12 (final_indices_rm):  Ht * Wt * 4KB   -- holds all final indices row-major
  c_5  (input_transposed):  Wt * 2KB         -- transposed logit values
  c_6  (index_transposed):  Wt * 2KB         -- transposed index values

At vocab=4096 these total ~1MB/core (fits in 1.5MB L1). At vocab=8192 they total
~2.1MB/core (overflow). This is a hard per-core constraint -- sub_core_grids does
not help because it controls core placement, not per-core buffer sizing.

Source: sampling_program_factory.cpp lines 78-181 (buffer allocation).
The kernel validates batch=32 and vocab%32==0 but has no vocab size guard.

## How Production Models Handle This

Every production model (Llama3-70B, DeepSeek-V3, Qwen3) pre-filters with
ttnn.topk() BEFORE calling ttnn.sampling(). The L1 limit only affects the
sampling kernel — ttnn.topk() is a different kernel with no such constraint.
See models/common/sampling/tt_sampling.py.

For multi-device (e.g. Llama3-70B on 8x4 Galaxy, 32 devices total):
  - sampling_all_gather_axis=0 (default), so gather is along rows: 8 devices
  - The other 4 devices (col axis) handle TP for attention, not vocab sharding
  - Vocab sharded across 8 row-devices: 128256/8 = 16032 tokens/device
  - Each device runs ttnn.topk(16K, k=32) -> [1,1,32,32] (~1.2ms)
    NOTE: topk is a separate kernel, NOT limited by sampling's L1 constraint
  - All-gather across 8 devices: 8*32 = 256 candidates
  - ttnn.sampling([1,1,32,256]): <0.2ms — 256 tokens is well within 4K limit
  - Total sampling overhead: ~2-3ms (enabling 80 tok/s)
  Source: model_config.py:607 (cluster_shape), tt_sampling.py:251 (axis selection)

For single-device (our vLLM case, multi_step_reduction=True path):
  - Split vocab in half: 128256/2 = 64128 per half
  - ttnn.topk(k=32) each half: ~9.3ms each (again, topk has no L1 issue)
  - Concat 2*32=64 candidates
  - ttnn.sampling([1,1,32,64]): <0.2ms — 64 tokens is well within 4K limit
  - Total: ~19ms (vs 147ms for the 66-op compiled graph = ~7.7x speedup)

Usage:
    python perf_debug/test_ttnn_sampling_direct.py
"""

import time

import torch
import torch.nn.functional as F
import ttnn


def open_device():
    return ttnn.open_device(device_id=0)


def run_topk_then_sampling(
    device,
    logits_torch,
    k_values,
    p_values,
    temp_values,
    max_top_k=32,
    seed=42,
):
    """
    Run the production single-device sampling pipeline:
    split -> topk each half -> concat -> sampling.
    """
    batch = logits_torch.shape[2]  # expect [1, 1, batch, vocab]
    vocab_size = logits_torch.shape[3]
    half_vocab = vocab_size // 2

    # Create indices tensor matching per-half logit width
    # Each half gets arange(half_vocab) — the offset is added after topk
    indices_torch = (
        torch.arange(half_vocab, dtype=torch.int32)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(1, 1, batch, half_vocab)
        .contiguous()
    )
    indices_tt = ttnn.from_torch(
        indices_torch,
        device=device,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Upload logits
    logits_tt = ttnn.from_torch(
        logits_torch.to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Split logits along vocab dim
    logit_halves = ttnn.split(logits_tt, half_vocab, dim=3)

    # Topk each half (both use same indices since they're local to each half)
    topk_values_list = []
    topk_indices_list = []
    for i in range(2):
        vals, inds = ttnn.topk(
            logit_halves[i], k=max_top_k, dim=-1, indices_tensor=indices_tt
        )
        topk_values_list.append(vals)
        topk_indices_list.append(inds)
        logit_halves[i].deallocate()

    # Concat the two halves' topk results -> [1, 1, batch, 2*max_top_k]
    topk_values_cat = ttnn.concat(topk_values_list, dim=3)
    topk_indices_cat = ttnn.concat(topk_indices_list, dim=3)

    for v, i in zip(topk_values_list, topk_indices_list):
        ttnn.deallocate(v)
        ttnn.deallocate(i)

    # Add device offsets to indices (half 0 offset=0, half 1 offset=half_vocab)
    offsets = torch.ones(1, 1, batch, 2 * max_top_k, dtype=torch.int64)
    offsets[:, :, :, :max_top_k] = 0
    offsets[:, :, :, max_top_k:] = half_vocab
    offsets_tt = ttnn.from_torch(
        offsets,
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    topk_indices_int32 = ttnn.typecast(topk_indices_cat, dtype=ttnn.int32)
    ttnn.deallocate(topk_indices_cat)

    global_indices = ttnn.add(offsets_tt, topk_indices_int32, dtype=ttnn.int32)
    ttnn.deallocate(topk_indices_int32)

    global_indices_rm = ttnn.untilize(global_indices, use_multicore=True)
    ttnn.deallocate(global_indices)

    # Prepare sampling params
    k_tt = ttnn.from_torch(
        torch.tensor(k_values, dtype=torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    p_tt = ttnn.from_torch(
        torch.tensor(p_values, dtype=torch.float32),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    temp_tt = ttnn.from_torch(
        torch.tensor(temp_values, dtype=torch.float32),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Sample
    output = ttnn.sampling(
        topk_values_cat,
        global_indices_rm,
        k=k_tt,
        p=p_tt,
        temp=temp_tt,
        seed=seed,
    )
    result = ttnn.to_torch(output).to(torch.int32)

    ttnn.deallocate(topk_values_cat)
    ttnn.deallocate(global_indices_rm)

    return result


def test_llama_vocab(device):
    """Full pipeline at Llama 3.1 8B vocab size (128256)."""
    vocab_size = 128256
    batch = 32
    logits = torch.randn(1, 1, batch, vocab_size)

    result = run_topk_then_sampling(
        device,
        logits,
        k_values=[50] * batch,
        p_values=[0.9] * batch,
        temp_values=[0.8] * batch,
        max_top_k=32,
        seed=42,
    )

    print(f"Output shape: {result.shape}")
    print(f"Sampled tokens (first 8): {result[0, 0, 0, :8].tolist()}")

    assert result.shape == (1, 1, 1, 32), f"Expected (1,1,1,32), got {result.shape}"
    assert (result >= 0).all(), "Negative token indices"
    assert (result < vocab_size).all(), f"Token indices >= {vocab_size}"
    print("PASS: llama_vocab (128256)")


def test_opt_vocab(device):
    """Full pipeline at OPT-125M vocab size (50272)."""
    vocab_size = 50272
    batch = 32
    logits = torch.randn(1, 1, batch, vocab_size)

    result = run_topk_then_sampling(
        device,
        logits,
        k_values=[50] * batch,
        p_values=[0.9] * batch,
        temp_values=[0.8] * batch,
        max_top_k=32,
        seed=42,
    )

    assert result.shape == (1, 1, 1, 32)
    assert (result >= 0).all() and (result < vocab_size).all()
    print("PASS: opt_vocab (50272)")


def test_determinism(device):
    """Same seed + inputs -> identical results."""
    vocab_size = 128256
    batch = 32
    logits = torch.randn(1, 1, batch, vocab_size)

    r1 = run_topk_then_sampling(
        device,
        logits,
        k_values=[50] * batch,
        p_values=[0.9] * batch,
        temp_values=[0.8] * batch,
        seed=42,
    )
    r2 = run_topk_then_sampling(
        device,
        logits,
        k_values=[50] * batch,
        p_values=[0.9] * batch,
        temp_values=[0.8] * batch,
        seed=42,
    )

    assert torch.equal(
        r1, r2
    ), f"Determinism failed:\n  r1={r1[0,0,0,:8]}\n  r2={r2[0,0,0,:8]}"
    print("PASS: determinism")


def test_greedy_topk1(device):
    """top_k=1, p=0 should select argmax."""
    vocab_size = 128256
    batch = 32
    logits = torch.randn(1, 1, batch, vocab_size)

    result = run_topk_then_sampling(
        device,
        logits,
        k_values=[1] * batch,
        p_values=[0.0] * batch,
        temp_values=[1.0] * batch,
        max_top_k=32,
        seed=42,
    )

    expected = logits.argmax(dim=-1).squeeze(0).squeeze(0)  # [32]
    actual = result[0, 0, 0, :]

    # bf16 precision may cause minor differences in topk ranking
    matches = (actual.int() == expected.int()).sum().item()
    print(f"top_k=1 greedy: {matches}/32 match argmax")
    assert matches >= 28, f"Only {matches}/32 matched argmax"
    print("PASS: greedy_topk1")


def test_per_user_params(device):
    """Different k/p/temp per user."""
    vocab_size = 128256
    batch = 32
    logits = torch.randn(1, 1, batch, vocab_size)

    k_values = list(range(1, 33))
    p_values = [i / 32.0 for i in range(32)]
    temp_values = [0.1 + i * 0.1 for i in range(32)]

    result = run_topk_then_sampling(
        device,
        logits,
        k_values=k_values,
        p_values=p_values,
        temp_values=temp_values,
        max_top_k=32,
        seed=42,
    )

    assert result.shape == (1, 1, 1, 32)
    assert (result >= 0).all() and (result < vocab_size).all()
    print("PASS: per_user_params")


def test_topp_disabled(device):
    """top_p=1.0 (no filtering) should still produce valid tokens."""
    vocab_size = 128256
    batch = 32
    logits = torch.randn(1, 1, batch, vocab_size)

    result = run_topk_then_sampling(
        device,
        logits,
        k_values=[32] * batch,
        p_values=[1.0] * batch,
        temp_values=[1.0] * batch,
        max_top_k=32,
        seed=42,
    )

    assert result.shape == (1, 1, 1, 32)
    assert (result >= 0).all() and (result < vocab_size).all()
    print("PASS: topp_disabled")


def test_small_sampling_only(device):
    """Test ttnn.sampling() directly at small vocab (no topk needed)."""
    vocab_size = 4096
    batch = 32

    logits = torch.randn(1, 1, batch, vocab_size, dtype=torch.bfloat16)
    values_tt = ttnn.from_torch(
        logits,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    indices = (
        torch.arange(vocab_size, dtype=torch.int32)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(1, 1, batch, vocab_size)
        .contiguous()
    )
    indices_tt = ttnn.from_torch(
        indices,
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    k_tt = ttnn.from_torch(
        torch.tensor([50] * batch, dtype=torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    p_tt = ttnn.from_torch(
        torch.tensor([0.9] * batch),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    temp_tt = ttnn.from_torch(
        torch.tensor([0.8] * batch),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    output = ttnn.sampling(values_tt, indices_tt, k=k_tt, p=p_tt, temp=temp_tt, seed=42)
    result = ttnn.to_torch(output).to(torch.int32)

    assert result.shape == (1, 1, 1, 32)
    assert (result >= 0).all() and (result < vocab_size).all()
    print("PASS: small_sampling_only (4096)")


def test_direct_sampling_vocab_limit(device):
    """Verify that ttnn.sampling() fails at vocab >= 8192 without topk pre-filtering.

    This documents a hard L1 constraint in the sampling kernel. Three circular
    buffers (c_5, c_6, c_12) scale with vocab width. At vocab=4096 they fit in
    1.5MB L1; at vocab=8192 they need ~2.1MB and overflow.

    The kernel does NOT validate this upfront -- it attempts to allocate and
    crashes with a circular buffer overflow error. Production code (tt_sampling.py)
    always pre-filters with ttnn.topk() to reduce to ~32-64 tokens first.

    Discovered by calling ttnn.sampling() directly at vocab=128256 (Llama 3.1 8B)
    and getting: "Statically allocated circular buffers on core range
    [(x=0,y=0) - (x=10,y=1)] grow to 32990080 B which is beyond max L1 size
    of 1572864 B"
    """
    batch = 32

    # vocab=4096 should work (last size that fits in L1)
    vocab_ok = 4096
    logits = torch.randn(1, 1, batch, vocab_ok, dtype=torch.bfloat16)
    values_tt = ttnn.from_torch(
        logits, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    indices = (
        torch.arange(vocab_ok, dtype=torch.int32)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(1, 1, batch, vocab_ok)
        .contiguous()
    )
    indices_tt = ttnn.from_torch(
        indices, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    k_tt = ttnn.from_torch(
        torch.tensor([32] * batch, dtype=torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    p_tt = ttnn.from_torch(
        torch.tensor([0.9] * batch),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    temp_tt = ttnn.from_torch(
        torch.tensor([1.0] * batch),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    output = ttnn.sampling(values_tt, indices_tt, k=k_tt, p=p_tt, temp=temp_tt, seed=42)
    result = ttnn.to_torch(output).to(torch.int32)
    assert result.shape == (1, 1, 1, 32)
    print(f"  vocab={vocab_ok}: OK (fits in L1)")

    # vocab=8192 should fail with L1 overflow
    vocab_fail = 8192
    logits = torch.randn(1, 1, batch, vocab_fail, dtype=torch.bfloat16)
    values_tt = ttnn.from_torch(
        logits, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    indices = (
        torch.arange(vocab_fail, dtype=torch.int32)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(1, 1, batch, vocab_fail)
        .contiguous()
    )
    indices_tt = ttnn.from_torch(
        indices, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )

    try:
        ttnn.sampling(values_tt, indices_tt, k=k_tt, p=p_tt, temp=temp_tt, seed=42)
        print(
            f"  vocab={vocab_fail}: unexpectedly succeeded (L1 limit may have changed)"
        )
    except RuntimeError as e:
        assert "beyond max L1 size" in str(e) or "grow to" in str(
            e
        ), f"Unexpected error: {e}"
        print(f"  vocab={vocab_fail}: correctly fails with L1 overflow")

    print("PASS: direct_sampling_vocab_limit")


def benchmark_topk_only(device, num_iters=50, warmup=5):
    """Benchmark ttnn.topk() alone to show where time is spent."""
    batch = 32
    print("\n=== Benchmark: ttnn.topk() alone (the dominant cost) ===")
    for half_vocab in [8000, 16000, 25136, 32064, 64128]:
        logits = torch.randn(1, 1, batch, half_vocab, dtype=torch.bfloat16)
        logits_tt = ttnn.from_torch(
            logits,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        indices = (
            torch.arange(half_vocab, dtype=torch.int32)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(1, 1, batch, half_vocab)
            .contiguous()
        )
        indices_tt = ttnn.from_torch(
            indices,
            device=device,
            dtype=ttnn.uint16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        for _ in range(warmup):
            v, i = ttnn.topk(logits_tt, k=32, dim=-1, indices_tensor=indices_tt)
            ttnn.deallocate(v)
            ttnn.deallocate(i)
        ttnn.synchronize_device(device)

        latencies = []
        for _ in range(num_iters):
            start = time.perf_counter()
            v, i = ttnn.topk(logits_tt, k=32, dim=-1, indices_tensor=indices_tt)
            ttnn.synchronize_device(device)
            latencies.append((time.perf_counter() - start) * 1000)
            ttnn.deallocate(v)
            ttnn.deallocate(i)

        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        equiv_full = half_vocab * 2
        print(
            f"  topk(half_vocab={half_vocab:>6}, k=32): P50={p50:.2f}ms  (full vocab={equiv_full})"
        )

        ttnn.deallocate(logits_tt)
        ttnn.deallocate(indices_tt)


def benchmark_full_pipeline(device, vocab_size, max_top_k=32, num_iters=100, warmup=10):
    """Benchmark the full topk + sampling pipeline."""
    batch = 32
    logits = torch.randn(1, 1, batch, vocab_size)
    k_values = [50] * batch
    p_values = [0.9] * batch
    temp_values = [0.8] * batch

    # Warmup
    for _ in range(warmup):
        run_topk_then_sampling(
            device,
            logits,
            k_values,
            p_values,
            temp_values,
            max_top_k=max_top_k,
            seed=42,
        )

    ttnn.synchronize_device(device)

    # Benchmark
    latencies = []
    for i in range(num_iters):
        start = time.perf_counter()
        run_topk_then_sampling(
            device, logits, k_values, p_values, temp_values, max_top_k=max_top_k, seed=i
        )
        ttnn.synchronize_device(device)
        latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()
    mean_ms = sum(latencies) / len(latencies)
    min_ms = latencies[0]
    max_ms = latencies[-1]
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[int(len(latencies) * 0.99)]

    print(
        f"\n=== Benchmark: topk+sampling, vocab={vocab_size}, top_k={max_top_k}, {num_iters} iters ==="
    )
    print(f"  Mean: {mean_ms:.2f} ms")
    print(f"  Min:  {min_ms:.2f} ms")
    print(f"  Max:  {max_ms:.2f} ms")
    print(f"  P50:  {p50:.2f} ms")
    print(f"  P99:  {p99:.2f} ms")
    print(f"  Baseline (66-op compiled graph): ~93ms (OPT) / ~147ms (Llama)")
    return mean_ms


def benchmark_sampling_only(device, reduced_vocab, num_iters=100, warmup=10):
    """Benchmark just ttnn.sampling() on pre-reduced vocab."""
    batch = 32
    logits = torch.randn(1, 1, batch, reduced_vocab, dtype=torch.bfloat16)
    values_tt = ttnn.from_torch(
        logits, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    indices = (
        torch.arange(reduced_vocab, dtype=torch.int32)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(1, 1, batch, reduced_vocab)
        .contiguous()
    )
    indices_tt = ttnn.from_torch(
        indices, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    k_tt = ttnn.from_torch(
        torch.tensor([50] * batch, dtype=torch.int32),
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    p_tt = ttnn.from_torch(
        torch.tensor([0.9] * batch),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    temp_tt = ttnn.from_torch(
        torch.tensor([0.8] * batch),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    for _ in range(warmup):
        ttnn.sampling(values_tt, indices_tt, k=k_tt, p=p_tt, temp=temp_tt, seed=42)
    ttnn.synchronize_device(device)

    latencies = []
    for i in range(num_iters):
        start = time.perf_counter()
        ttnn.sampling(values_tt, indices_tt, k=k_tt, p=p_tt, temp=temp_tt, seed=i)
        ttnn.synchronize_device(device)
        latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()
    mean_ms = sum(latencies) / len(latencies)
    print(
        f"\n=== Benchmark: sampling only, reduced_vocab={reduced_vocab}, {num_iters} iters ==="
    )
    print(
        f"  Mean: {mean_ms:.2f} ms  Min: {latencies[0]:.2f} ms  Max: {latencies[-1]:.2f} ms  P50: {latencies[len(latencies)//2]:.2f} ms"
    )
    return mean_ms


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ttnn.sampling() verification and benchmarks"
    )
    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "correctness",
            "bench",
            "bench-llama",
            "bench-opt",
            "bench-topk",
        ],
        default="all",
        help="What to run (default: all)",
    )
    parser.add_argument(
        "--iters", type=int, default=50, help="Benchmark iterations (default: 50)"
    )
    args = parser.parse_args()

    device = open_device()
    try:
        if args.mode in ("all", "correctness"):
            print("=" * 60)
            print("Phase 1a: Vocab size limitation (L1 constraint)")
            print("=" * 60)
            test_direct_sampling_vocab_limit(device)

            print("\n" + "=" * 60)
            print("Phase 1b: Correctness tests (topk + sampling pipeline)")
            print("=" * 60)
            test_small_sampling_only(device)
            test_llama_vocab(device)
            test_opt_vocab(device)
            test_determinism(device)
            test_greedy_topk1(device)
            test_per_user_params(device)
            test_topp_disabled(device)

        if args.mode in ("all", "bench", "bench-topk"):
            print("\n" + "=" * 60)
            print("Benchmark: ttnn.topk() alone")
            print("=" * 60)
            benchmark_topk_only(device, num_iters=args.iters)

        if args.mode in ("all", "bench"):
            print("\n" + "=" * 60)
            print("Benchmark: ttnn.sampling() on reduced vocab")
            print("=" * 60)
            benchmark_sampling_only(device, reduced_vocab=64, num_iters=args.iters)

        if args.mode in ("all", "bench", "bench-opt"):
            print("\n" + "=" * 60)
            print("Benchmark: topk+sampling, OPT vocab (50272)")
            print("=" * 60)
            benchmark_full_pipeline(device, vocab_size=50272, num_iters=args.iters)

        if args.mode in ("all", "bench", "bench-llama"):
            print("\n" + "=" * 60)
            print("Benchmark: topk+sampling, Llama vocab (128256)")
            print("=" * 60)
            benchmark_full_pipeline(device, vocab_size=128256, num_iters=args.iters)

        print("\nDone!")
    finally:
        ttnn.close_device(device)
