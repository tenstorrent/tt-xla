# Plan: Sub-1ms Non-Greedy Sampling on Device

Tracking issue: https://github.com/tenstorrent/tt-xla/issues/3940

All measurements at Llama 3.1 8B vocab (128,256), single-device Wormhole (Blackhole), batch=32.

## Summary of Findings

We discovered that non-greedy sampling can run in **0.89ms on device** — matching greedy argmax — by exploiting multi-core `ttnn.topk()`. The key is padding vocab chunks to power-of-2 sizes under 65536, which triggers the multi-core bitonic sort path (uses all available cores instead of 1).

| Path | Latency | Cores Used | Status |
|---|---|---|---|
| Current device (66-op compiled graph) | ~147ms | mixed (cumsum: 1 core) | ships today |
| Single-core topk + sampling (naive) | 18.76ms | topk: 1 core | measured |
| **Multi-core topk + sampling** | **0.89ms** | **topk: 110 cores** | **measured, not integrated** |
| Greedy argmax (device) | <1ms | multi-core | baseline target |
| Current CPU sampling | 4.7-133ms | N/A (batch dependent) | ships today on kmabee/vllm_demo |

## Phase 1: Verify ttnn.sampling() on hardware (DONE)

Completed 2026-03-26. See `perf_debug/test_ttnn_sampling_direct.py`.

Key findings:
- `ttnn.sampling()` overflows L1 at vocab >= 8192 (per-core buffer constraint in `sampling_program_factory.cpp`)
- Production models always pre-filter with `ttnn.topk()` — sampling only sees ~32-256 tokens
- All correctness tests pass: determinism, greedy (32/32 argmax match), per-user params, edge cases

## Phase 2: Performance benchmarking (DONE)

Completed 2026-03-26/27. Tracy profiling confirmed device kernel times.

### Multi-core topk discovery (2026-03-27)

Tracy profiling revealed `ttnn.topk` running on **1 core out of 110**. Investigation of the topk kernel source (`topk_device_operation.cpp`, `topk_utils.cpp`, `topk_constants.hpp`) found that multi-core topk IS fully implemented but gated behind:

1. **Input dimension must be power of 2** (bitonic sort algorithm)
2. **Input dimension must be < 65536** (uint16 index range)
3. **k must be <= 64**
4. **Work must divide evenly across cores**

Our vocab (128,256) fails conditions 1 and 2. Fix: split into 4 chunks of 32064, pad each to 32768 (largest power of 2 under 65536).

| Approach | TopK Cores | P50 Latency |
|---|---|---|
| topk(128256) single call | 1 | 19.85ms |
| 2x topk(64128) split-in-half | 1 each | 18.76ms |
| **4x topk(32768) padded** | **110 each** | **0.89ms** |
| topk(32768) single call | 110 | 0.18ms |
| topk(65536) | 1 (falls back) | 10.21ms |

Tracy op breakdown for one iteration of the multi-core pipeline (from `ops_perf_results` CSV):

| Op | Cores | Device Time |
|---|---|---|
| Slice ×4 (split chunks) | 110 | 0.02ms each |
| Pad ×4 (to 32768) | — | <0.01ms each |
| **TopK ×4 (multi-core)** | **110** | **0.18ms each** |
| Concat ×2 | 2 | <0.01ms |
| Typecast (uint16→int32) | 2 | <0.01ms |
| Add (global offsets) | 110 | <0.01ms |
| Untilize | 1 | <0.01ms |
| **Sampling** | **32** | **0.03ms** |

Note: the 46ms number reported earlier was inflated by host-side tensor recreation every iteration. With persistent tensors the single-core path was 18.76ms wall-clock, matching Tracy's 18.4ms device time. The multi-core path is 0.89ms wall-clock.

## Phase 3: Integration approach (REVISED)

### Original plan: custom op (paged_update_cache pattern)

The original plan was to register `tt::sampling` as a custom op following the `paged_update_cache` pattern, with compiler lowering through tt-mlir. However:

- Custom ops in tt-xla map to **single ttnn ops** at runtime (confirmed by reading `paged_update_cache.cpp` runtime dispatch). The runtime calls one `ttnn::` function, not a sequence.
- Our pipeline is a **sequence** of ttnn ops (split → pad → topk ×4 → concat → offset-add → untilize → sampling). This can't be a single custom op without creating a new fused kernel in tt-metal.
- Creating a new fused ttnn kernel is significant tt-metal work and defeats the purpose of reusing existing ops.

### New approach: rewrite sampler torch code (compile-through)

Since `torch.compile(backend="tt")` compiles torch ops → tt-mlir → ttnn ops, we can rewrite the sampler to use torch ops that lower to the right ttnn ops with multi-core-friendly shapes:

```python
# In sampler.py, replace the current 66-op sampling logic with:
# 1. Pad logits to next multiple of chunk size
# 2. Split into chunks of 32768 (power of 2, < 65536)
# 3. torch.topk(k=32) on each chunk → compiles to multi-core ttnn.topk
# 4. Concat candidates
# 5. Apply top-p / temperature / Gumbel-max on the small candidate set
```

This approach:
- Requires **no custom op registration** in tt-xla
- Requires **no tt-mlir compiler changes**
- Compiles through the existing `torch.compile(backend="tt")` path
- Multi-core topk triggers automatically because input shapes are power-of-2 and < 65536

**Risk**: the compiler must preserve the power-of-2 shapes through to `ttnn.topk`. If any intermediate pass reshapes or pads differently, multi-core won't trigger. Needs validation.

**Where the padding logic lives**: in `integrations/vllm_plugin/vllm_tt/sampler.py`, as regular torch ops in the compiled function. The pad-to-power-of-2 and chunk-splitting are explicit torch operations that the compiler traces and lowers. This matches how the existing sampler already does explicit torch ops (sort, softmax, etc.) inside the compiled graph.

### Fallback: custom op for ttnn.sampling() only

If the compile-through approach doesn't preserve multi-core shapes, fall back to:
- Use the compile-through path for topk (the expensive part)
- Register a custom op only for `ttnn.sampling()` (the cheap 0.03ms part that has the L1 vocab limit)
- This is simpler than the original Phase 3+4 plan since the custom op is just the sampling op, not the full pipeline

### Can the existing 66-op graph be improved?

The current 66-op graph's bottlenecks (from original Tracy profiling):
- Accumulation (cumsum): 24ms on **1 core** — same single-core problem as topk
- FillPad (ttnn.full ×11): 14ms on **1 core**
- Sort: 14ms on 110 cores
- Softmax ×3: 7ms on 110 cores

The multi-core topk discovery suggests: **if we replaced the sort+cumsum+softmax+multinomial sequence with topk on power-of-2 padded chunks, the existing compiled graph would go from ~93ms to potentially sub-1ms.** The cumsum (24ms, 1 core) and FillPad (14ms, 1 core) are entirely eliminated because topk replaces them.

This is essentially the same as the "rewrite sampler torch code" approach above — we're replacing the expensive ops with topk, which happens to have an excellent multi-core implementation when shapes are right.

## Phase 3: Validation results (2026-03-30)

### Composite op lowering confirmed working

After rebasing onto commits including tt-mlir#7504 and tt-xla#3729, tested on new Ubuntu 24.04 container (clang-20):

**`torch.topk` now compiles to `TopKDeviceOperation`, not `SortDeviceOperation`.**

Tracy results from `perf_debug/test_compiled_topk_multicore.py`:

| Op | Cores | Kernel time | Previous (sort) |
|---|---|---|---|
| `TopKDeviceOperation` (32768-wide chunk) | **65** | **0.181ms** | 6.11ms (Sort, 110 cores) |
| `SortDeviceOperation` | **0** (gone) | — | present before |

The composite op pass replaces `torch.topk` → `composite_topk` → `stablehlo.composite @tenstorrent.topk` → `ttnn.topk`. Multi-core triggers at 32768 (power of 2, < 65536).

Direct ttnn benchmark (persistent tensors, confirming no regression):
- 4-way split multicore topk + sampling: **P50 = 0.90ms** (matching pre-build baseline of 0.89ms)

Note: wall-clock through `torch.compile` path is ~2.1ms P50 due to 4 separate kernel dispatches vs the single-call direct ttnn case. Tracy kernel time (0.181ms × 4 = ~0.72ms) is the real device cost.

**Next step**: rewrite `sampler.py` to use `torch.topk` on padded 32768-wide chunks and verify the full vLLM compiled sampling path gets multi-core topk.

## Phase 4: Implementation plan

### Step 1: Rewrite sampler torch code

**File**: `integrations/vllm_plugin/vllm_tt/sampler.py`

Replace the current `apply_top_k_top_p` + softmax + Gumbel-max sequence with:
1. Pad logits to multiple of 32768
2. Split into chunks of 32768
3. `torch.topk(k=32)` per chunk (compiles to multi-core ttnn.topk)
4. Concat all chunk results → [batch, num_chunks × 32] candidates
5. Add per-chunk offsets to get global vocab indices
6. Apply top-p on the small candidate set
7. Temperature scaling + Gumbel-max on the small candidate set

### Step 2: Validate compiler preserves shapes

Compile the new sampler and check Tracy output:
- Confirm `TopKDeviceOperation` shows CORE_COUNT > 1
- Confirm input dimensions are 32768 (power of 2)

### Step 3: Correctness validation

- Greedy (temp=0) produces identical tokens vs current path
- Non-greedy with same seed produces same distribution
- Run existing sampling tests

### Step 4: Performance validation

Target: < 1ms total sampling time at Llama vocab, any batch size up to 32.

## CPU Sampling Optimizations (back pocket)

Documented in `perf_debug/cpu_sampling_optimization.md`. Two code fixes to `sample_from_logits_cpu` (on `kmabee/vllm_demo` branch) for immediate deployment if device path is blocked:

1. **Batch topk-first before top-p**: Sort only k elements instead of 128K. Exact equivalence when top_k > 0.
2. **Gumbel-max trick**: Replace softmax+multinomial with argmax+gumbel_noise. Exact equivalence.
3. **Config: enable top_k=50**: Makes fixes 1+2 maximally effective.

Expected gains with all three (Llama vocab):

| Batch | Current CPU | Optimized CPU |
|---|---|---|
| 1 | 8.7ms | 0.5ms |
| 8 | 73ms | 0.7ms |
| 16 | 138ms | ~1ms |
| 32 | 342ms | 6.3ms |

These are easier to deploy today (pure Python changes, no compiler work) and serve as fallback if the device integration hits issues.

## Key files

| File | Role |
|---|---|
| `integrations/vllm_plugin/vllm_tt/sampler.py` | Sampler rewrite target |
| `integrations/vllm_plugin/vllm_tt/model_runner.py` | Decode loop, sample_from_logits |
| `perf_debug/test_ttnn_sampling_direct.py` | Verification tests and benchmarks |
| `perf_debug/cpu_sampling_optimization.md` | CPU fallback optimization analysis |
| `ttnn/.../operations/reduction/topk/device/topk_device_operation.cpp` | Multi-core topk gate logic |
| `ttnn/.../operations/reduction/topk/device/topk_constants.hpp` | `multi_core_min_width = 8192` |
| `ttnn/.../operations/reduction/topk/device/topk_utils.cpp` | `find_topk_core_config()` |
