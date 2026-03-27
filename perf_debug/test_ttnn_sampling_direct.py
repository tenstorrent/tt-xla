# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone test for ttnn.topk() + ttnn.sampling() pipeline at production vocab sizes.

Key finding: ttnn.sampling() can only handle small vocab sizes (~4096 max) due to
L1 buffer constraints. Production models (DeepSeek, Llama3-70B) always pre-filter
with ttnn.topk() first, reducing vocab to ~32-64 tokens before sampling.

For single-device, the production pattern (from tt_sampling.py) is:
  1. Split logits in half along vocab dim
  2. ttnn.topk() each half -> max_top_k results per half
  3. Concat -> 2 * max_top_k candidates
  4. ttnn.sampling() on the reduced set

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
    device = open_device()
    try:
        print("=" * 60)
        print("Phase 1: Correctness tests")
        print("=" * 60)

        test_small_sampling_only(device)
        test_llama_vocab(device)
        test_opt_vocab(device)
        test_determinism(device)
        test_greedy_topk1(device)
        test_per_user_params(device)
        test_topp_disabled(device)

        print("\n" + "=" * 60)
        print("Phase 2: Performance benchmarks")
        print("=" * 60)

        # Sampling-only benchmark (reduced vocab after topk)
        benchmark_sampling_only(device, reduced_vocab=64, num_iters=100)

        # Full pipeline benchmarks
        benchmark_full_pipeline(device, vocab_size=50272, num_iters=50)
        benchmark_full_pipeline(device, vocab_size=128256, num_iters=50)

        print("\nAll tests passed!")
    finally:
        ttnn.close_device(device)
