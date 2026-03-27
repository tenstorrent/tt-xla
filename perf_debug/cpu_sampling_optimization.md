# CPU Sampling Optimization for vLLM on TT-XLA

Tracking issue: https://github.com/tenstorrent/tt-xla/issues/3940

All numbers measured at Llama 3.1 8B vocab size (128,256) on single-device Wormhole.

## Branch Context

The CPU sampling path being optimized here lives on `kmabee/vllm_demo` (not merged to main). It was put together quickly when device sampling was found to be slow for non-greedy:

- `0edf092de` (2026-03-21): Fix cpu_sampling path never being reached in sample_tokens
- `0c7ffadac` (2026-03-22): Extend CPU sampling to support temperature, penalties, and top-k/top-p

These commits added the `sample_from_logits_cpu` method that this document analyzes. The proposed fixes build on that implementation.

## Current State

Non-greedy sampling has three paths today:

| Path | Batch 1 | Batch 32 | Status |
|---|---|---|---|
| Device (66-op compiled graph) | ~147ms | ~147ms | default (`cpu_sampling=False`) |
| CPU sampling | 4.7ms | 133ms | opt-in (`cpu_sampling=True`) |
| Greedy argmax (device) | <1ms | <1ms | when `temperature=0` |

CPU sampling is already the faster non-greedy path at low batch sizes. The goal is to close the gap with greedy (<1ms) and maintain that advantage at higher batch sizes.

## Bottleneck Analysis

### With current demo settings (temp=0.6, top_k=0, top_p=1.0)

Per-step breakdown at batch=1, vocab=128256:

| Step | Time | Notes |
|---|---|---|
| bf16->f32 cast | 0.03ms | |
| Repetition penalty | 0.05ms | Elementwise, scales with batch |
| Temperature scaling | 0.02ms | Elementwise |
| `torch.softmax` (128K) | 0.15ms | |
| **`torch.multinomial` (128K)** | **4.07ms** | Builds cumulative distribution over all 128K tokens |
| **Total** | **~4.8ms** | |

Bottleneck: `torch.multinomial` on the full 128K vocab.

### With top_k=50, top_p=0.9

Per-step breakdown at batch=1, vocab=128256:

| Step | Time | Notes |
|---|---|---|
| Temperature scaling | 0.35ms | |
| Top-k (k=50) | 0.55ms | |
| **Top-p (full `torch.sort` on 128K)** | **15.71ms** | Sorts all 128K tokens including 128K-50 that are already -inf |
| `torch.softmax` + `torch.multinomial` (128K) | 5.23ms | |
| **Total** | **~22ms** | |

Bottleneck: `torch.sort` on 128K elements for top-p, even though top-k already reduced the candidate set to 50 tokens.

## Proposed Fixes

### Fix 1: Batch topk-first before top-p

When `top_k > 0`, apply `torch.topk` on the full batch as a single batched op, then run top-p sort/cumsum on only k elements. The current code applies top-k by masking logits to -inf, then sorts the full 128K vocab for top-p — wastefully sorting 128K-50 dead values.

This is mathematically exact (verified: identical probability distributions, 0.00e+00 max difference across all 128K tokens, same 43 active tokens in both paths over 10K sampling trials).

**Applies when**: `top_k > 0` and `top_p < 1.0`

### Fix 2: Gumbel-max trick

Replace `torch.softmax(logits) + torch.multinomial(probs)` with `torch.argmax(logits + gumbel_noise)`. This is a well-known exact equivalence for sampling from a categorical distribution. The device-compiled sampler already uses this trick (`sampler.py` lines 194-197).

Eliminates both the softmax pass and the cumulative-distribution search in multinomial. Works on the full batch as a single `argmax` op.

**Applies to**: all non-greedy sampling

### Fix 3 (config recommendation): Enable top_k=50

Not a code change. With top_k=50, fixes 1+2 reduce the sort/sample working set from 128K to 50 elements, making CPU sampling nearly constant-cost across batch sizes.

Current demo defaults (temp=0.6, top_k=0, top_p=1.0) were chosen quickly to avoid garbage output. Adding top_k=50 with top_p=0.9 is standard practice and filters low-probability noise tokens.

## Expected Gains

### With current settings (temp=0.6, top_k=0, top_p=1.0)

Fix 2 only (fix 1 doesn't apply since top_k=0):

| Batch | Current CPU | After fix 2 | Savings |
|---|---|---|---|
| 1 | 4.7ms | 2.4ms | 2.3ms |
| 8 | 30ms | 10.7ms | 19ms |
| 16 | 59ms | 25ms | 34ms |
| 32 | 133ms | 68ms | 65ms |

### With top_k=50 added (temp=0.6, top_k=50, top_p=0.9)

All three fixes:

| Batch | Current CPU | After fixes 1+2+3 | Savings |
|---|---|---|---|
| 1 | 8.7ms | 0.5ms | 8.2ms |
| 8 | 73ms | 0.7ms | 72ms |
| 16 | 138ms | ~1ms | 137ms |
| 32 | 342ms | 6.3ms | 336ms |

## Comparison Across All Paths

All paths at Llama 3.1 8B vocab (128,256), single-device Wormhole:

| Path | Batch 1 | Batch 32 | Status |
|---|---|---|---|
| Current device (66-op compiled graph) | ~147ms | ~147ms | ships today |
| Proposed device (ttnn.topk + ttnn.sampling) | ~46ms | ~46ms | not yet implemented |
| Current CPU sampling | 4.7ms | 133ms | ships today with `cpu_sampling=True` |
| Proposed CPU (fixes 1+2, no top_k) | 2.4ms | 68ms | not yet implemented |
| Proposed CPU (fixes 1+2+3, top_k=50) | 0.5ms | 6.3ms | not yet implemented |
| Greedy argmax (device) | <1ms | <1ms | baseline target |

With top_k=50, the optimized CPU path beats the device path (both current and proposed) at every batch size. Without top_k, the proposed device path (46ms) wins over optimized CPU (68ms) at batch=32.

## Why the Device Path is Slow on Single-Chip

The proposed device path uses `ttnn.topk()` to reduce vocab before `ttnn.sampling()` (which overflows L1 at vocab >= 8192). The topk kernel must scan all 128K tokens and takes ~9.3ms per half-vocab. On single-chip, the full vocab lives on one device. On multi-chip (e.g., Llama 70B on 8x4 Galaxy), the vocab is sharded across 8 devices — each only scans 16K tokens (~1.2ms) in parallel.

See `perf_debug/ttnn_sampling_integration_plan.md` for the full device integration plan and `perf_debug/test_ttnn_sampling_direct.py` for the L1 limitation tests.

## Files to Modify

| File | Change |
|---|---|
| `integrations/vllm_plugin/vllm_tt/model_runner.py` | Fix 1+2 in `sample_from_logits_cpu` (~line 2181) |
| Demo/deployment config | Fix 3: set `top_k=50, top_p=0.9` |

## Implementation Notes

- Fix 1 eliminates per-request `for` loops (lines 2220-2240) with batched `torch.topk` + batched sort on k elements
- Fix 2 replaces lines 2243-2247 (`softmax + multinomial + where`) with `argmax(logits + gumbel_noise)`
- Both fixes are independent and can be shipped separately
- The Gumbel-max trick requires `logits` (not probabilities) as input, which is what `sample_from_logits_cpu` already works with
