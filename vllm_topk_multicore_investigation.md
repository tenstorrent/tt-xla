# ttnn.topk Multi-core Investigation: Why Sampling is Slow at Batch=1

## Summary

The `test_sampling_op_overhead.py` benchmark showed 4×chunked topk taking **14.81ms** at
batch=1, while production Tracy captures showed **0.18ms per topk** (65 cores). The
`sampling_b32` test (batch=32 input) shows **1.09ms** total for 4×topk + ttnn.sampling.
This documents the root cause and implications.

---

## Root Cause

**Kernel efficiency scales with batch (tensor height).** For small tensors (batch=1),
hardware cores are underutilized. For large tensors (batch=32), all 65 cores run in
parallel giving near-maximum throughput.

| Test | Input batch | Ops | Time | Core utilization |
|---|---|---|---|---|
| `greedy` | 1 | few | 0.23ms | n/a (argmax) |
| `topk_only` | 1 | 34 | 14.81ms | few cores per op |
| `sampling` (pad-to-32 inside) | 1→32 | 36 | 14.86ms | slow ops run BEFORE pad |
| **`sampling_b32`** | **32** | **28** | **1.09ms** | 65+ cores, all ops fast |
| `standalone` (eager sampling) | 32 | - | 1.53ms | sampling kernel only |
| Tracy model (65 cores) | 32 | - | **0.18ms/topk** | fully utilized |

---

## Why Pad-to-32 Inside the Graph Doesn't Help

The original pad-to-32 optimization (`86a1210b1`) pads [1, vocab] → [32, vocab] **inside**
the compiled graph. But the graph already ran many batch=1 ops (slice, typecast, etc.)
before reaching the topk. Those ops run slowly because small tensors don't fill the cores.

| Phase | Without pad-to-32 | With pad-to-32 (inside graph) |
|---|---|---|
| 4× slice(logits→chunk) | batch=1, slow | batch=1, slow |
| 4× pad(chunk→pow2) | batch=1, slow | batch=1, slow |
| 4× typecast(f32→bf16) | batch=1, slow | batch=1→32, fast |
| 4× topk | batch=1, slow | batch=32, fast |
| ttnn.sampling | batch=32 | batch=32 |

Only the topk and sampling benefit. The slow ops are still slow.

**Pad-to-32 must happen BEFORE the compiled graph** (on the input logits) to benefit
all operations.

---

## Additional Issue: Pad-to-32 Causes Quality Regression with ttnn.sampling

When pad-to-32 is applied to the input logits (batch=1 → batch=32 with -inf dummy rows),
the batched topk on [32, vocab] returns **correct values** (cosine_sim=1.0) but the
**indices for row 0 may differ** from single-batch topk [1, vocab]. This causes
ttnn.sampling to output garbage tokens.

Workaround: `pad_to_batch=False` for the ttnn.sampling path (current code).

Fix needed: investigate why batched topk with -inf dummy rows returns different index
sets for row 0. May be a ttnn.topk hardware issue with mixed real/dummy rows.

---

## Production Timing (with trace)

In production vLLM with `enable_trace=True`, the sampling graph is captured once and
replayed. The dispatch overhead is eliminated and only kernel execution matters:

| Op | Kernel time |
|---|---|
| 4× topk (65 cores, [1, 32768]) | 4 × 0.18ms = 0.72ms |
| ttnn.sampling ([32, 128]) | ~0.18ms |
| Other ops | ~0.1ms |
| **Total** | **~1ms per token** |

This confirms the original 17% speedup claim was real: the sampling graph with trace
takes ~1ms, and pad-to-32 reduces topk cost from ~3ms to 0.72ms per token.

---

## Standalone Benchmark Recommendations

The `test_sampling_op_overhead.py` benchmark is only accurate for **production-like**
conditions when:
1. Input is `batch=32` (pre-padded), OR
2. The benchmark measures `sampling_b32` mode

The `topk_only` and `sampling` (batch=1) modes measure the **non-trace, non-batched**
path which overstates overhead vs production.

`sampling_b32` at **1.09ms** is the correct production baseline for the sampling graph.

---

## Next Steps

1. Fix the pad-to-32 + ttnn.sampling quality issue (wrong indices from batched topk)
   so both optimizations can work together
2. Investigate whether `pad_to_batch=True` with ttnn.sampling can be made safe (e.g.
   by sorting candidates after batched topk to ensure consistent value-index pairing)
3. Profile `sampling_b32` with Tracy to confirm core counts match the model's behavior
