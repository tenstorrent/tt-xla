#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Isolate ttnn.sampling overhead layer by layer.

Each test adds one layer on top of the previous:
  greedy     → argmax only (baseline)
  topk_only  → 4x chunked topk, no sampling
  topk_pad   → topk + pad to batch=32, no sampling
  sampling   → topk + pad + ttnn.sampling (full path)
  standalone → ttnn.sampling only, pre-shaped inputs

Deltas between adjacent tests show where time goes:
  topk_only - greedy    = topk overhead
  topk_pad - topk_only  = padding overhead
  sampling - topk_pad   = ttnn.sampling in compiled graph
  standalone             = ttnn.sampling kernel + runtime overhead

Usage:
  TT_USE_TTNN_SAMPLING=1 python3 perf_debug/test_sampling_op_overhead.py all
  TT_USE_TTNN_SAMPLING=1 python3 perf_debug/test_sampling_op_overhead.py sampling
  TT_USE_TTNN_SAMPLING=1 python3 perf_debug/test_sampling_op_overhead.py topk_only
  TT_USE_TTNN_SAMPLING=1 python3 perf_debug/test_sampling_op_overhead.py topk_pad
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

VOCAB = 128256
CHUNK = 32768
K_PER_CHUNK = 32


def benchmark(fn, args, name, warmup=5, iters=20):
    import torch_xla

    for _ in range(warmup):
        fn(*args)
        torch_xla.sync()
    t0 = time.perf_counter()
    out = None
    for _ in range(iters):
        out = fn(*args)
    # Materialize result to ensure device work is complete.
    result = out[0] if isinstance(out, tuple) else out
    result.cpu()
    t1 = time.perf_counter()
    ms = (t1 - t0) / iters * 1000
    print(f"  {name}: {ms:.2f} ms/call  ({iters} iters)")
    return ms


def _topk_body(logits):
    """Shared 4x chunked topk logic."""
    chunks = torch.split(logits, CHUNK, dim=-1)
    all_vals, all_inds = [], []
    for i, c in enumerate(chunks):
        if c.shape[-1] < CHUNK:
            c = torch.nn.functional.pad(
                c, (0, CHUNK - c.shape[-1]), value=float("-inf")
            )
        v, idx = torch.topk(c, k=K_PER_CHUNK, dim=-1)
        all_vals.append(v)
        all_inds.append(idx + i * CHUNK)
    vals = torch.cat(all_vals, dim=-1)
    inds = torch.cat(all_inds, dim=-1)
    return vals, inds


def run_greedy():
    """Argmax only — baseline."""
    dev = xm.xla_device()

    @torch.compile(backend="tt", fullgraph=True, dynamic=False)
    def graph(logits):
        return logits.argmax(dim=-1).view(-1)

    logits = torch.randn(1, VOCAB, dtype=torch.float32).to(dev)
    print("=== greedy: argmax (batch=1, vocab=128256) ===")
    return benchmark(graph, (logits,), "greedy")


def run_topk_only():
    """4x chunked topk, return values+indices. No sampling."""
    dev = xm.xla_device()

    @torch.compile(backend="tt", fullgraph=True, dynamic=False)
    def graph(logits):
        vals, inds = _topk_body(logits)
        return vals.to(torch.bfloat16), inds.to(torch.int32)

    logits = torch.randn(1, VOCAB, dtype=torch.float32).to(dev)
    print("=== topk_only: 4x topk (batch=1, vocab=128256) ===")
    return benchmark(graph, (logits,), "topk_only")


def run_topk_pad():
    """4x topk + pad to batch=32. No sampling."""
    dev = xm.xla_device()

    @torch.compile(backend="tt", fullgraph=True, dynamic=False)
    def graph(logits, k_in, p_in, temp_in):
        vals, inds = _topk_body(logits)
        vals = vals.to(torch.bfloat16)
        inds = inds.to(torch.int32)
        vals = torch.nn.functional.pad(vals, (0, 0, 0, 31), value=float("-inf"))
        inds = torch.nn.functional.pad(inds, (0, 0, 0, 31))
        k_in = torch.nn.functional.pad(k_in, (0, 31), value=1)
        p_in = torch.nn.functional.pad(p_in, (0, 31), value=1.0)
        temp_in = torch.nn.functional.pad(temp_in, (0, 31), value=1.0)
        # Return all padded tensors to prevent dead-code elimination
        return vals, inds, k_in, p_in, temp_in

    logits = torch.randn(1, VOCAB, dtype=torch.float32).to(dev)
    k = torch.full((1,), 32, dtype=torch.int32).to(dev)
    p = torch.ones(1, dtype=torch.bfloat16).to(dev)
    temp = torch.full((1,), 1.667, dtype=torch.bfloat16).to(dev)
    print("=== topk_pad: 4x topk + pad to batch=32 ===")
    return benchmark(graph, (logits, k, p, temp), "topk_pad")


def run_sampling():
    """4x topk + pad + ttnn.sampling (full path)."""
    dev = xm.xla_device()

    @torch.compile(backend="tt", fullgraph=True, dynamic=False)
    def graph(logits, k_in, p_in, temp_in):
        vals, inds = _topk_body(logits)
        vals = vals.to(torch.bfloat16)
        inds = inds.to(torch.int32)
        vals = torch.nn.functional.pad(vals, (0, 0, 0, 31), value=float("-inf"))
        inds = torch.nn.functional.pad(inds, (0, 0, 0, 31))
        k_in = torch.nn.functional.pad(k_in, (0, 31), value=1)
        p_in = torch.nn.functional.pad(p_in, (0, 31), value=1.0)
        temp_in = torch.nn.functional.pad(temp_in, (0, 31), value=1.0)
        return torch.ops.tt.sampling(vals, inds, k_in, p_in, temp_in, 42)

    logits = torch.randn(1, VOCAB, dtype=torch.float32).to(dev)
    k = torch.full((1,), 32, dtype=torch.int32).to(dev)
    p = torch.ones(1, dtype=torch.bfloat16).to(dev)
    temp = torch.full((1,), 1.667, dtype=torch.bfloat16).to(dev)
    print("=== sampling: 4x topk + pad + ttnn.sampling ===")
    return benchmark(graph, (logits, k, p, temp), "sampling")


def run_standalone():
    """ttnn.sampling only, pre-shaped [32,128] inputs."""
    dev = xm.xla_device()
    vals = torch.randn(32, 128, dtype=torch.bfloat16).to(dev)
    indices = torch.randint(0, 128000, (32, 128), dtype=torch.int32).to(dev)
    k = torch.full((32,), 32, dtype=torch.int32).to(dev)
    p = torch.ones(32, dtype=torch.bfloat16).to(dev)
    temp = torch.ones(32, dtype=torch.bfloat16).to(dev)

    print("=== standalone: tt.sampling only (pre-shaped, no topk) ===")
    return benchmark(
        lambda v, i, k, p, t: (torch.ops.tt.sampling(v, i, k, p, t, 42),),
        (vals, indices, k, p, temp),
        "standalone",
    )


TESTS = {
    "greedy": run_greedy,
    "topk_only": run_topk_only,
    "topk_pad": run_topk_pad,
    "sampling": run_sampling,
    "standalone": run_standalone,
}

# Order for "all" — layered from simplest to most complex
ALL_ORDER = ["greedy", "topk_only", "topk_pad", "sampling", "standalone"]


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which == "all":
        results = {}
        for name in ALL_ORDER:
            results[name] = TESTS[name]()
            print()

        print("=== Summary ===")
        for name in ALL_ORDER:
            print(f"  {name:15s}: {results[name]:.2f} ms")
        print()
        print("=== Deltas ===")
        pairs = list(zip(ALL_ORDER, ALL_ORDER[1:]))
        for prev, curr in pairs:
            if curr == "standalone":
                continue  # standalone is a different path, not additive
            delta = results[curr] - results[prev]
            print(f"  {prev:15s} → {curr:15s}: +{delta:.2f} ms")
    elif which in TESTS:
        TESTS[which]()
    else:
        print(f"Unknown test: {which}")
        print(f"Usage: {sys.argv[0]} [{' | '.join(['all'] + list(TESTS))}]")
        sys.exit(1)


if __name__ == "__main__":
    import torch_xla

    main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Sync device before exit so tracy can capture all pending ops.
    # If this hangs, fall back to: os._exit(0)
    torch_xla.sync()
    sys.exit(0)
