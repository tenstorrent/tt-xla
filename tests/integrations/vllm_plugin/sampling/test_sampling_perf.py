# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Benchmark sampling graph performance: device vs CPU, greedy vs non-greedy.

Uses synthetic logits (no model loading) so iteration is fast. Compiles the
sampling graph once, then measures steady-state per-sample latency over many
iterations. Results are collected into a JSON file via perf_collector.

Usage:
    # Run all benchmarks (default 100 iterations each)
    pytest -svv tests/integrations/vllm_plugin/sampling/test_sampling_perf.py

    # Quick run with fewer iterations
    pytest -svv tests/integrations/vllm_plugin/sampling/test_sampling_perf.py --iterations 20

    # Custom output path
    pytest -svv tests/integrations/vllm_plugin/sampling/test_sampling_perf.py --perf-output perf_debug/my_run.json

    # Single config
    pytest -svv tests/integrations/vllm_plugin/sampling/test_sampling_perf.py -k "greedy_device"

    # Filter by vocab size
    pytest -svv tests/integrations/vllm_plugin/sampling/test_sampling_perf.py -k "llama3"
"""

import time

import pytest
import torch
from vllm_tt.metadata import XLASupportedSamplingMetadata
from vllm_tt.sampler import Sampler


@pytest.fixture
def device():
    import torch_xla.core.xla_model as xm

    return xm.xla_device()


def make_metadata(
    batch_size=1,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    min_p=0.0,
    all_greedy=False,
    device=None,
):
    dev = device or torch.device("cpu")
    return XLASupportedSamplingMetadata(
        temperature=torch.full((batch_size,), temperature, device=dev),
        top_k=torch.full((batch_size,), top_k, dtype=torch.int32, device=dev),
        top_p=torch.full((batch_size,), top_p, device=dev),
        min_p=torch.full((batch_size,), min_p, device=dev),
        all_greedy=all_greedy,
    )


def run_sampler(logits, metadata):
    sampler = Sampler()
    return sampler(logits, metadata).sampled_token_ids


def run_sampler_greedy(logits, metadata):
    return torch.argmax(logits, dim=-1, keepdim=True)


def run_sampler_cpu(logits, metadata):
    """Run the same Sampler on CPU (no torch.compile)."""
    sampler = Sampler()
    return sampler(logits, metadata).sampled_token_ids


def compile_and_warmup(fn, logits, metadata, label):
    """Compile via torch.compile and run once to trigger compilation."""
    compiled_fn = torch.compile(fn, backend="tt", dynamic=False)
    print(f"\n  Compiling {label} ...", flush=True)
    start = time.perf_counter()
    compiled_fn(logits, metadata)
    elapsed = time.perf_counter() - start
    print(f"  Compilation took {elapsed:.1f}s", flush=True)
    return compiled_fn


def benchmark_sampling(fn, logits, metadata, iterations, warmup=5):
    """Run fn for warmup + iterations, return per-call ms stats."""
    for _ in range(warmup):
        fn(logits, metadata)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn(logits, metadata)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    times.sort()
    # Drop top/bottom 5% for stability
    trim = max(1, len(times) // 20)
    trimmed = times[trim:-trim] if len(times) > 2 * trim else times

    avg = sum(trimmed) / len(trimmed)
    p50 = trimmed[len(trimmed) // 2]
    p95 = trimmed[int(len(trimmed) * 0.95)]
    return avg, p50, p95


def report(label, avg, p50, p95, iterations):
    tok_s = 1000.0 / avg if avg > 0 else 0
    print(f"\n  {label}")
    print(f"    iterations: {iterations}")
    print(f"    avg: {avg:.2f} ms/sample  ({tok_s:.1f} tok/s)")
    print(f"    p50: {p50:.2f} ms")
    print(f"    p95: {p95:.2f} ms")


VOCAB_SIZES = [
    pytest.param(50272, id="opt_125m"),
    pytest.param(128256, id="llama3"),
]


# -- Device sampling benchmarks --


@pytest.mark.single_device
@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
def test_greedy_device(device, vocab_size, iterations, perf_collector):
    """Benchmark: greedy (argmax) compiled on device."""
    logits = torch.randn(1, vocab_size, dtype=torch.float32).to(device)

    compiled_fn = compile_and_warmup(
        run_sampler_greedy, logits, None, f"greedy_device[{vocab_size}]"
    )

    avg, p50, p95 = benchmark_sampling(compiled_fn, logits, None, iterations)
    report(f"GREEDY DEVICE (vocab={vocab_size})", avg, p50, p95, iterations)
    perf_collector(
        "greedy_device", "device", "greedy", vocab_size, iterations, avg, p50, p95
    )


@pytest.mark.single_device
@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
def test_non_greedy_device(device, vocab_size, iterations, perf_collector):
    """Benchmark: non-greedy (temp=0.8, top_p=0.9) compiled on device."""
    logits = torch.randn(1, vocab_size, dtype=torch.float32).to(device)
    metadata = make_metadata(temperature=0.8, top_k=50, top_p=0.9, device=device)

    compiled_fn = compile_and_warmup(
        run_sampler, logits, metadata, f"non_greedy_device[{vocab_size}]"
    )

    avg, p50, p95 = benchmark_sampling(compiled_fn, logits, metadata, iterations)
    report(f"NON-GREEDY DEVICE (vocab={vocab_size})", avg, p50, p95, iterations)
    perf_collector(
        "non_greedy_device",
        "device",
        "non_greedy",
        vocab_size,
        iterations,
        avg,
        p50,
        p95,
    )


@pytest.mark.single_device
@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
def test_temperature_only_device(device, vocab_size, iterations, perf_collector):
    """Benchmark: temperature only, no top-p filtering (top_p=1.0, top_k=vocab)."""
    logits = torch.randn(1, vocab_size, dtype=torch.float32).to(device)
    metadata = make_metadata(
        temperature=0.8, top_k=vocab_size, top_p=1.0, device=device
    )

    compiled_fn = compile_and_warmup(
        run_sampler, logits, metadata, f"temp_only_device[{vocab_size}]"
    )

    avg, p50, p95 = benchmark_sampling(compiled_fn, logits, metadata, iterations)
    report(f"TEMP-ONLY DEVICE (vocab={vocab_size})", avg, p50, p95, iterations)
    perf_collector(
        "temp_only_device", "device", "temp_only", vocab_size, iterations, avg, p50, p95
    )


# -- CPU sampling benchmarks --


@pytest.mark.single_device
@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
def test_non_greedy_cpu(vocab_size, iterations, perf_collector):
    """Benchmark: non-greedy sampling entirely on CPU (no device, no compile)."""
    logits = torch.randn(1, vocab_size, dtype=torch.float32)
    metadata = make_metadata(temperature=0.8, top_k=50, top_p=0.9)

    avg, p50, p95 = benchmark_sampling(run_sampler_cpu, logits, metadata, iterations)
    report(f"NON-GREEDY CPU (vocab={vocab_size})", avg, p50, p95, iterations)
    perf_collector(
        "non_greedy_cpu", "cpu", "non_greedy", vocab_size, iterations, avg, p50, p95
    )


@pytest.mark.single_device
@pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
def test_greedy_cpu(vocab_size, iterations, perf_collector):
    """Benchmark: greedy (argmax) on CPU."""
    logits = torch.randn(1, vocab_size, dtype=torch.float32)

    avg, p50, p95 = benchmark_sampling(run_sampler_greedy, logits, None, iterations)
    report(f"GREEDY CPU (vocab={vocab_size})", avg, p50, p95, iterations)
    perf_collector("greedy_cpu", "cpu", "greedy", vocab_size, iterations, avg, p50, p95)
