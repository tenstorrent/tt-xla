#!/usr/bin/env python3
"""Reproduce SamplingOp per-call overhead matching vLLM 8B decode shapes.

Measures the compiled graph execution time for the sampling path:
  logits[1, 128256] → 4x topk(32768) → pad to batch=32 → ttnn.sampling

Run:
  TT_USE_TTNN_SAMPLING=1 python3 perf_debug/test_sampling_op_overhead.py
"""

import os
import time

os.environ.setdefault("TT_USE_TTNN_SAMPLING", "1")

import torch
import torch_xla.core.xla_model as xm

from tt_torch.custom_ops import sampling  # noqa: F401 — registers the op


def make_sampling_graph():
    """Compiled graph matching the vLLM non-greedy sampler path."""

    @torch.compile(backend="tt", fullgraph=True, dynamic=False)
    def graph(logits, k_in, p_in, temp_in):
        # 4x chunked topk (matching _TOPK_MAX_CHUNK_SIZE=32768, k=32/chunk)
        chunks = torch.split(logits, 32768, dim=-1)
        all_vals, all_inds = [], []
        for i, c in enumerate(chunks):
            if c.shape[-1] < 32768:
                c = torch.nn.functional.pad(
                    c, (0, 32768 - c.shape[-1]), value=float("-inf")
                )
            v, idx = torch.topk(c, k=32, dim=-1)
            all_vals.append(v)
            all_inds.append(idx + i * 32768)
        vals = torch.cat(all_vals, dim=-1).to(torch.bfloat16)
        inds = torch.cat(all_inds, dim=-1).to(torch.int32)

        # Pad batch 1→32 for ttnn.sampling
        vals = torch.nn.functional.pad(vals, (0, 0, 0, 31), value=float("-inf"))
        inds = torch.nn.functional.pad(inds, (0, 0, 0, 31))
        k_in = torch.nn.functional.pad(k_in, (0, 31), value=1)
        p_in = torch.nn.functional.pad(p_in, (0, 31), value=1.0)
        temp_in = torch.nn.functional.pad(temp_in, (0, 31), value=1.0)

        return torch.ops.tt.sampling(vals, inds, k_in, p_in, temp_in, 42)

    return graph


def make_greedy_graph():
    """Compiled greedy graph for comparison (argmax only, no topk/sampling)."""

    @torch.compile(backend="tt", fullgraph=True, dynamic=False)
    def graph(logits):
        return logits.argmax(dim=-1).view(-1)

    return graph


def benchmark(fn, args, name, warmup=5, iters=20):
    for _ in range(warmup):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    t1 = time.perf_counter()
    ms = (t1 - t0) / iters * 1000
    print(f"{name}: {ms:.2f} ms/call  ({iters} iters)")
    return ms


def main():
    dev = xm.xla_device()

    # 8B decode shapes
    logits = torch.randn(1, 128256, dtype=torch.float32).to(dev)
    k = torch.full((1,), 32, dtype=torch.int32).to(dev)
    p = torch.ones(1, dtype=torch.bfloat16).to(dev)
    temp = torch.full((1,), 1.667, dtype=torch.bfloat16).to(dev)  # 1/0.6

    sampling_graph = make_sampling_graph()
    greedy_graph = make_greedy_graph()

    print("=== Llama-3.1-8B decode shapes (batch=1, vocab=128256) ===")
    ms_sampling = benchmark(
        sampling_graph, (logits, k, p, temp), "4x topk + sampling"
    )
    ms_greedy = benchmark(greedy_graph, (logits,), "greedy (argmax)")
    print(f"Overhead: {ms_sampling - ms_greedy:.2f} ms ({ms_sampling/ms_greedy:.1f}x)")

    # Also test standalone sampling (no topk) to isolate the op
    vals = torch.randn(32, 128, dtype=torch.bfloat16).to(dev)
    indices = torch.randint(0, 128000, (32, 128), dtype=torch.int32).to(dev)
    k32 = torch.full((32,), 32, dtype=torch.int32).to(dev)
    p32 = torch.ones(32, dtype=torch.bfloat16).to(dev)
    temp32 = torch.ones(32, dtype=torch.bfloat16).to(dev)

    print("\n=== Standalone ttnn.sampling (no topk, pre-shaped inputs) ===")
    benchmark(
        lambda v, i, k, p, t: (
            torch.ops.tt.sampling(v, i, k, p, t, 42),
            xm.mark_step(),
        ),
        (vals, indices, k32, p32, temp32),
        "tt.sampling only",
    )


if __name__ == "__main__":
    main()
