#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Reproduce SamplingOp per-call overhead matching vLLM 8B decode shapes.

Usage:
  TT_USE_TTNN_SAMPLING=1 python3 perf_debug/test_sampling_op_overhead.py all
  TT_USE_TTNN_SAMPLING=1 python3 perf_debug/test_sampling_op_overhead.py sampling
  TT_USE_TTNN_SAMPLING=1 python3 perf_debug/test_sampling_op_overhead.py greedy
  TT_USE_TTNN_SAMPLING=1 python3 perf_debug/test_sampling_op_overhead.py standalone
"""

import os
import sys
import time

os.environ.setdefault("TT_USE_TTNN_SAMPLING", "1")

import torch
import torch_xla.core.xla_model as xm
from tt_torch.custom_ops import sampling  # noqa: F401 — registers the op


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


def run_sampling():
    """4x topk + pad + ttnn.sampling (batch=1, vocab=128256)."""
    dev = xm.xla_device()

    @torch.compile(backend="tt", fullgraph=True, dynamic=False)
    def graph(logits, k_in, p_in, temp_in):
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
        vals = torch.nn.functional.pad(vals, (0, 0, 0, 31), value=float("-inf"))
        inds = torch.nn.functional.pad(inds, (0, 0, 0, 31))
        k_in = torch.nn.functional.pad(k_in, (0, 31), value=1)
        p_in = torch.nn.functional.pad(p_in, (0, 31), value=1.0)
        temp_in = torch.nn.functional.pad(temp_in, (0, 31), value=1.0)
        return torch.ops.tt.sampling(vals, inds, k_in, p_in, temp_in, 42)

    logits = torch.randn(1, 128256, dtype=torch.float32).to(dev)
    k = torch.full((1,), 32, dtype=torch.int32).to(dev)
    p = torch.ones(1, dtype=torch.bfloat16).to(dev)
    temp = torch.full((1,), 1.667, dtype=torch.bfloat16).to(dev)

    print("=== 4x topk + sampling (batch=1, vocab=128256) ===")
    return benchmark(graph, (logits, k, p, temp), "4x topk + sampling")


def run_greedy():
    """Greedy argmax baseline (same vocab shape)."""
    dev = xm.xla_device()

    @torch.compile(backend="tt", fullgraph=True, dynamic=False)
    def graph(logits):
        return logits.argmax(dim=-1).view(-1)

    logits = torch.randn(1, 128256, dtype=torch.float32).to(dev)

    print("=== Greedy argmax (batch=1, vocab=128256) ===")
    return benchmark(graph, (logits,), "greedy (argmax)")


def run_standalone():
    """Standalone tt.sampling only (pre-shaped [32,128] inputs)."""
    dev = xm.xla_device()
    vals = torch.randn(32, 128, dtype=torch.bfloat16).to(dev)
    indices = torch.randint(0, 128000, (32, 128), dtype=torch.int32).to(dev)
    k = torch.full((32,), 32, dtype=torch.int32).to(dev)
    p = torch.ones(32, dtype=torch.bfloat16).to(dev)
    temp = torch.ones(32, dtype=torch.bfloat16).to(dev)

    print("=== Standalone tt.sampling (pre-shaped, no topk) ===")
    return benchmark(
        lambda v, i, k, p, t: (
            torch.ops.tt.sampling(v, i, k, p, t, 42),
            xm.mark_step(),
        ),
        (vals, indices, k, p, temp),
        "tt.sampling only",
    )


TESTS = {
    "sampling": run_sampling,
    "greedy": run_greedy,
    "standalone": run_standalone,
}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which == "all":
        ms_s = run_sampling()
        print()
        ms_g = run_greedy()
        print(f"Overhead vs greedy: {ms_s - ms_g:.2f} ms ({ms_s/ms_g:.1f}x)")
        print()
        run_standalone()
    elif which in TESTS:
        TESTS[which]()
    else:
        print(f"Unknown test: {which}")
        print(f"Usage: {sys.argv[0]} [{' | '.join(['all'] + list(TESTS))}]")
        sys.exit(1)


if __name__ == "__main__":
    main()
