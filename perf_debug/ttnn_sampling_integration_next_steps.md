# Integrating ttnn.sampling() as a Custom Op in tt-xla

## Context

Non-greedy sampling on device for Llama-3.1-8B is 4.4x slower than greedy (5.1 vs 22.2 tok/s). We've identified the bottleneck and have a partial fix that gets to 12.3 tok/s, but there's still a 45% gap to greedy. This document describes the next step to close that gap.

## What we've proven so far

### 1. The sort-based sampling graph is the bottleneck
The compiled non-greedy sampling graph in `sampler.py` uses `probs.sort()` on the full 128K vocab, producing `SortDeviceOperation` (30ms) + `AccumulationDeviceOperation` (24ms, 1 core) + `FillPadDeviceOperation` (14ms, 1 core) = ~60ms/token overhead.

### 2. Multi-core topk pre-filtering works and is essentially free
We replaced the sort with `torch.topk` on 4 power-of-2 padded chunks (4 × 32768), which compiles to multi-core `ttnn.topk` (65 cores, 0.18ms/chunk) via the composite op lowering (tt-xla#3729, tt-mlir#7504). **Experiment proved the 4x topk adds zero overhead to greedy** (18.6 → 19.1 tok/s, within noise).

### 3. The remaining bottleneck is post-topk sampling ops
After topk reduces to 64 candidates, the remaining ops — softmax, exponential noise generation, div, argmax, gather back to global index — compiled via torch.compile add ~6ms/token. This is the gap between greedy (18.6) and non-greedy with topk fix (12.3).

### 4. Sampling on the reduced candidate set has a correctness bug
Doing softmax → Gumbel-max → gather(candidate_indices, local_idx) on the 64-candidate set produces incorrect token IDs when compiled in the full sampler graph. It works in standalone compiled tests but fails within the vLLM sampler context. Root cause unknown — likely a compiler/runtime issue with the gather op in the larger graph.

### 5. ttnn.sampling() is fast and already exists
We benchmarked `ttnn.sampling()` directly (calling ttnn Python API, not through torch.compile):
- 0.03ms on 64 tokens, 32 cores
- Does softmax + top-k filter + top-p filter + multinomial in one fused kernel
- Used in production tt-metal models (Llama-70B, DeepSeek, Qwen3)

## What ttnn.sampling() would give us

```
Current non-greedy (baseline):
  logits → sort(128K) → cumsum → softmax → sample     = 5.1 tok/s

Current fix (v1 topk, correct output workaround):
  logits → 4x topk(32K) → scatter(128K) → softmax(128K) → sample  = 10.9 tok/s

With ttnn.sampling():
  logits → 4x topk(32K) → ttnn.sampling(64 candidates)            ≈ 18+ tok/s (projected)
```

The 4x topk is free (proven). ttnn.sampling() is 0.03ms (proven). The integration work is plumbing the custom op through the compiler so it can be called from the compiled torch graph.

## What needs to be done

### Step 1: Register custom op in tt-xla

**File**: `python_package/tt_torch/custom_ops.py`

Follow the `paged_update_cache` pattern (line ~453 in that file). Register `tt::sampling`:

```python
@torch.library.custom_op("tt::sampling", mutates_args=[], device_types=["xla", "cpu"])
def sampling(input_values, input_indices, k, p, temp, seed: int):
    # XLA path: stablehlo_custom_call with "tt.sampling" target
    # CPU fallback: reference Python implementation
```

Inputs (all from the topk output, already on device):
- `input_values`: [batch, 64] bf16 — the topk'd logit values
- `input_indices`: [batch, 64] int32 — global vocab indices for each candidate
- `k`: [batch] uint32 — per-user top-k
- `p`: [batch] bf16 — per-user top-p
- `temp`: [batch] bf16 — per-user temperature
- `seed`: int — random seed

Output:
- `[batch]` int32 — one sampled global token index per user

Register `@sampling.register_fake` for shape inference.

### Step 2: Add compiler lowering in tt-mlir

Follow the `paged_update_cache` lowering pattern across these files:

1. **TTIR op definition**: `include/ttmlir/Dialect/TTIR/IR/TTIROps.td` — add `TTIR_SamplingOp`
2. **TTNN op definition**: `include/ttmlir/Dialect/TTNN/IR/TTNNOps.td` — add `TTNN_SamplingOp`
3. **StableHLO → TTIR**: `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp` — match `funcName == "tt.sampling"`
4. **TTIR → TTNN**: `lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`
5. **Flatbuffer schema**: new `include/ttmlir/Target/TTNN/operations/sampling.fbs`
6. **Flatbuffer serialization**: `lib/Target/TTNN/TTNNToFlatbuffer.cpp`
7. **Runtime dispatch**: new `runtime/lib/ttnn/operations/sampling/sampling.cpp` — calls `ttnn::sampling()`
8. **EmitC/EmitPy**: add patterns in `TTNNToEmitC.cpp` and `TTNNToEmitPy.cpp`

### Step 3: Wire into sampler.py

In `integrations/vllm_plugin/vllm_tt/sampler.py`, the `sample()` method's non-greedy path becomes:

```python
# 4x topk pre-filtering (already working, compiles to multi-core ttnn.topk)
filtered_logits, candidate_indices = apply_top_k_top_p_fast(logits, ...)

# Replace: softmax → Gumbel-max → gather (broken, ~6ms)
# With: single custom op call (~0.03ms)
sampled_token = torch.ops.tt.sampling(
    filtered_logits, candidate_indices,
    sampling_metadata.top_k, sampling_metadata.top_p,
    sampling_metadata.temperature, seed
)
```

This bypasses three problems at once:
- **Correctness bug**: no gather needed — ttnn.sampling returns global token IDs directly
- **Performance**: one kernel (~0.03ms) vs many compiled ops (~6ms)
- **Scatter regression**: no scatter back to full vocab needed (the v2 workaround used scatter which killed batch=32 performance, going from 3.8 baseline to 1.5 tok/s). ttnn.sampling() takes the reduced candidate set directly.

## Key constraints

### ttnn.sampling() batch dimension must be 32
The kernel hardcodes `N*C*H == 32` users. For batch<32, pad to 32 and extract results. For batch=32, maps directly. Source: `sampling_device_operation.cpp:47`.

### ttnn.sampling() L1 limit: vocab < ~4096
The kernel overflows L1 at vocab ≥ 8192. This is fine because we feed it the 64-candidate reduced set from topk, not the full vocab. Source: `sampling_program_factory.cpp:78-181`.

### Input format requirements
- `input_values`: bf16, TILE layout
- `input_indices`: int32, ROW_MAJOR layout
- `k`: uint32, ROW_MAJOR
- `p`: bf16, ROW_MAJOR
- `temp`: bf16, ROW_MAJOR

## tt-metal source for ttnn.sampling()

```
ttnn/cpp/ttnn/operations/reduction/sampling/
├── sampling.hpp/cpp                    # Public API
├── sampling_nanobind.hpp/cpp           # Python bindings
└── device/
    ├── sampling_device_operation.hpp/cpp  # Validation, constraints
    ├── sampling_program_factory.hpp/cpp   # GPU program setup
    └── kernels/compute/sampling.cpp       # Compute kernel
```

Existing tests:
- `tests/ttnn/unit_tests/operations/eltwise/test_sampling.py`
- `models/common/tests/test_sampling.py`

## Performance targets

| Config | Current | With ttnn.sampling() | Greedy (target) |
|---|---|---|---|
| Batch=1 | 5.1 tok/s (baseline) / 12.3 (v1 fix) | ~18+ tok/s | 22.2 tok/s |
| Batch=32 | 4.5 tok/s (baseline) / 10.7 (v1 fix) | ~14+ tok/s | 14.2 tok/s |

## Reference files

| File | Purpose |
|---|---|
| `integrations/vllm_plugin/vllm_tt/sampler.py` | Where the sampling call lives |
| `python_package/tt_torch/custom_ops.py` | Custom op registration (paged_update_cache as template) |
| `perf_debug/sampling_perf_apr12.md` | Full performance investigation notes |
| `perf_debug/ttnn_sampling_integration_plan.md` | Earlier detailed integration plan |
| `perf_debug/test_ttnn_sampling_direct.py` | Direct ttnn.sampling() tests and benchmarks |

## Status: Integration complete, correctness bug open (Apr 14)

### What was done
- Full tt-mlir pipeline: TTIR/TTNN SamplingOp, StableHLO→TTIR→TTNN conversions, flatbuffer, runtime, EmitC/EmitPy, OpModel stubs (21 files, commit `ffc4810ce` on `kmabee/apr12_vllm_demo_sampling_op_integration`)
- tt-xla: `tt::sampling` custom op in `custom_ops.py`, sampler wiring behind `TT_USE_TTNN_SAMPLING=1`, OPT-125M benchmark tests (commit `f42bdc362` on `kmabee/vllm_perf_apr12`)
- Runtime handles: 2D→4D reshape, ROW_MAJOR layout enforcement, UINT32↔INT32 typecast
- Batch=1→32 padding workaround for ttnn.sampling kernel's batch=32 constraint

### Performance results (correctness fixed, commit `9b7bb1431`)
| Config | Without ttnn.sampling | With ttnn.sampling | Greedy |
|---|---|---|---|
| OPT-125M b1 | 9.1 tok/s | ~9 tok/s | 11.3 tok/s |
| Llama-3.1-8B b1 | 10.9 tok/s | **12.6 tok/s** | ~19 tok/s |

### Correctness bug: FIXED
**Root cause:** Double temperature + wrong convention. The sampler applied `logits / temperature` before topk, then passed raw `temperature` to ttnn.sampling which multiplies by it (kernel expects `1/temperature`, per tt-metal `generator.py:452`). Net effect: temperature canceled out, wrong softmax distribution.

**Fix:** Skip `apply_temperature` in the sampler when using ttnn.sampling (the kernel handles it). Pass `1/temperature` as the temp parameter. For greedy (temp~0), use temp_recip=1.0 and k=1 (matching production convention). This also improved perf (10.7 → 12.6 tok/s for 8B) by eliminating the full-vocab temperature divide.

### Op count overhead analysis
The compiled non-greedy graph has 62 non-bookkeeping TTNN ops vs 50 for greedy+topk. The 12 extra ops are: temperature divide on full 128K vocab, 6 padding ops (batch 1→32), torch.where+gt (greedy/random merge), and 6 extra typecasts. The 19 typecasts total (vs 13 in greedy) are the largest single category.

### Tracy profiling results (OPT-125M, Apr 14)

Per-call SamplingOp overhead breakdown from tracy:
- **SamplingOp: 16.6 ms/call** (runtime: 2× reshape + 4× to_layout + 2× typecast + kernel)
- **TopKOp: 7.2 ms/call** × 2 chunks on OPT vocab = 14.4 ms total
- **PadOp: 1.9 ms/call** × several = ~10 ms
- Total sampling overhead: **~45 ms/token** vs ~50 ms model forward

The 16.6ms SamplingOp is dominated by runtime conversions (to_layout for 4 inputs), not the kernel itself (0.03ms). Attempted removing runtime to_layout calls but the workarounds pass doesn't reliably enforce ROW_MAJOR → kernel crashes. The workarounds need to be fixed at the compiler level.

### Root cause analysis (Apr 14)

The compiler workarounds pass correctly sets ROW_MAJOR for sampling inputs (confirmed via MLIR dump). The runtime `to_layout` calls are no-ops when layout is already correct. The 16.6ms SamplingOp time is from **5 device dispatches** inside the runtime:
1. `ttnn::reshape` input_values (2D→4D)
2. `ttnn::reshape` input_indices (2D→4D)
3. `ttnn::typecast` k (INT32→UINT32)
4. `ttnn::reshape` output (4D→1D)
5. `ttnn::typecast` output (UINT32→INT32)

Each device dispatch has ~2-3ms host-device round-trip on Blackhole. The ttnn::sampling kernel itself is 0.03ms.

### Performance improvement opportunities
1. **Eliminate runtime typecast for k** — make the compiler insert INT32→UINT32 typecast in the TTNN workarounds pass so it's baked into the compiled program. This removes 1 device dispatch (~3ms).
2. **Eliminate runtime typecast for output** — make the compiler expect UINT32 output from SamplingOp and insert UINT32→INT32 after. Removes 1 dispatch (~3ms).
3. **Eliminate runtime reshapes** — either make the kernel accept 2D input directly, or reshape in the compiler. Removes 2 dispatches (~5ms).
4. **Batch=32** — kernel's natural batch size. No padding needed, all tensors already aligned.
5. **Reduce topk chunks** — 2 chunks for smaller vocabs (OPT 50K fits in 2×32K).

### Trace mode: trisc1 firmware compiler ICE (Apr 14)

Trace-enabled benchmarks (`enable_trace=True`) crash with a segfault in the RISC-V cross-compiler during LTO of the trisc1 firmware:
```
lto1: internal compiler error: Segmentation fault
riscv-tt-elf-g++ (sfpi:7.32.0[333]) 15.1.0
```
Reproducer: `TT_USE_TTNN_SAMPLING=1 pytest -svv "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison_trace[8b-b1-nongreedy-device-trace]"`

This is a tt-metal sfpi toolchain bug, not a ttnn.sampling issue. Trace mode compiles different kernel programs than non-trace (for capture/replay), and the resulting firmware binary triggers an LTO ICE in GCC 15.1.0. Non-trace mode works fine.

Cache path with the failing build: `/home/kmabee/.cache/tt-metal-cache/15296582388382065472/firmware/trisc1/`

**Update:** After clearing `~/.cache/tt-metal-cache/`, ALL firmware builds crashed with LTO ICE. Root cause: sfpi toolchain was 7.32.0, tt-metal requires 7.39.0. Fixed by upgrading sfpi (`install_dependencies.sh --sfpi`). Not specific to ttnn.sampling.

**Tensor::reshape experiment (Apr 14):** Attempted using `Tensor::reshape` (metadata-only view) instead of `ttnn::reshape` in the runtime. Tracy showed it's **slower** (28.1 ms/call vs 16.6 ms/call). `Tensor::reshape` on TILE-layout tensors likely triggers internal data movement. Reverted to `ttnn::reshape`.

**Llama-3.1-8B tracy profile (Apr 14, steady-state with warmup):**
- SamplingOp: 34ms/call (3.4% of total), 7 calls
- TopKOp: 3.6ms/call (1.5%), 29 calls
- PadOp: 1.1ms/call (1.0%), 63 calls
- Sampling overhead = ~5.8% of total runtime, but 43% of per-token time at 12.7 tok/s
- Biggest overall bottleneck: LoadCachedOp (27.6%) and ToDeviceOp (15.2%) — program cache and data transfer overhead

### Standalone overhead breakdown (Apr 14)

Isolated each layer of the sampling path using `perf_debug/test_sampling_op_overhead.py`:

```
perf_debug/sampling_overhead_logs/1_standalone.log:  standalone: 0.18 ms/call  (20 iters)
perf_debug/sampling_overhead_logs/2_greedy.log:  greedy: 0.15 ms/call  (20 iters)
perf_debug/sampling_overhead_logs/3_topk_only.log:  topk_only: 1.14 ms/call  (20 iters)
perf_debug/sampling_overhead_logs/4_topk_pad.log:  topk_pad: 1.34 ms/call  (20 iters)
perf_debug/sampling_overhead_logs/5_sampling.log:  sampling: 1.49 ms/call  (20 iters)
```

Deltas:
- greedy → topk_only: +0.99 ms (4x chunked topk — dominant cost, 66% of total)
- topk_only → topk_pad: +0.20 ms (padding 5 tensors to batch=32)
- topk_pad → sampling: +0.15 ms (ttnn.sampling op itself — essentially free)
- standalone tt.sampling: 0.18 ms (kernel + runtime reshape/typecast — matches greedy)

**Total standalone overhead vs greedy: 1.34 ms.** The sampling path is near-optimal in isolation. The 26ms gap in the full 8B model (12.7 vs ~19 tok/s) is not from sampling ops — it's from other differences between greedy and non-greedy model graphs (penalties, graph shape, program dispatch overhead).

### Full model results summary (Apr 14)

| Config | tok/s | Notes |
|---|---|---|
| Greedy (current build) | 19.0 | baseline, no sampling ops |
| Greedy (pre-sampling build) | 20.4 | from apr12_sampling_comparison/ — 7% faster, likely tt-mlir version diff |
| Non-greedy + ttnn.sampling, no penalty | 13.4 | temp=0.6 only |
| Non-greedy + ttnn.sampling, with penalty | 12.7 | temp=0.6 + rep_penalty=1.1 |
| Non-greedy + ttnn.sampling, with penalty + trace | 13.8 | trace enabled |

Gap analysis (vs 19.0 greedy on current build):
- Penalties: 0.7 tok/s (13.4 → 12.7) — small, only 12% of gap
- Trace: +1.1 tok/s (12.7 → 13.8) — moderate
- **Remaining unexplained gap: 5.6 tok/s (19.0 → 13.4)**

The standalone overhead test shows the sampling path adds only 1.34ms over greedy. At 19 tok/s (53ms/token), adding 1.34ms would give ~18.5 tok/s — not 13.4. The 5.6 tok/s gap (~22ms/token) is 16× larger than the standalone overhead.

**Root cause hypothesis:** The non-greedy sampler code changes the compiled decode graph structure (more ops, different control flow). The compiler optimizes the combined model+sampler graph differently — potentially worse memory placement, less op fusion, or different program dispatch scheduling. This affects the entire decode step, not just the sampling portion.

### Key experiments (Apr 14)

| Experiment | tok/s | What it tests |
|---|---|---|
| Greedy baseline | 19.0 | Pure argmax |
| TT_GREEDY_WITH_SAMPLING_OPS | 18.7 | **Flawed** — sampling ops were dead code (all_greedy=True → argmax fast path, sampler never called) |
| TT_FORCE_SAMPLING_METADATA | 18.8 | Metadata CPU→device transfers — NOT the bottleneck |
| TT_NONGREEDY_ARGMAX_ONLY | **18.6** | Non-greedy config + argmax only — proves non-greedy model_runner path is fast |
| Non-greedy + ttnn.sampling, no penalty | **13.4** | Full topk+sampling in sampler |

**Root cause:** The torch.compile'd sampler graph (Program B, `all_greedy=False`) is ~22ms/token slower when it includes topk+sampling ops vs just argmax. The same ops add only 1.34ms in a standalone compiled graph. The compiler generates a much worse XLA program when topk+sampling are part of the sampler module's torch.compile scope.

This is a **torch.compile / XLA graph optimization issue**, not a sampling op performance issue.

### Compiled sampler graph comparison (Apr 15)

Extracted TTNN IR for the non-greedy vs greedy sampler compiled programs:

**Greedy sampler (graph 10): 3 ops** — to_layout, argmax, typecast
**Non-greedy sampler (graph 9): 58 ops** — breakdown:
- 17 typecast, 9 pad, 5 slice, 5 full, 4 topk, 3 to_layout, 3 add, 2 concat
- 1 sampling, 1 where, 1 gt, 1 divide, 1 reshape

The 55 extra ops each have dispatch overhead (~0.3-1ms per op in the vLLM execution context). Total: ~22ms extra per decode step, matching the observed gap.

**Fix: reduce op count in the sampler graph.** Main targets:
- 17 typecasts — many are from bf16↔f32↔int32 conversions for padding/topk
- 9 pads — batch-1→32 padding for ttnn.sampling kernel
- 5 slices + 5 fulls — padding infrastructure

### Proof checklist — does reducing ops improve tok/s?

- [x] **Experiment A**: Non-greedy sampler with topk ONLY (no pad, no sampling) → **13.2 tok/s**. Topk alone causes the entire slowdown. Padding and ttnn.sampling add nothing. The 4x topk adds 0.99ms standalone but ~25ms in the vLLM sampler compiled program — a 25x amplification.
- [x] **Experiment A2**: Single topk (not 4x chunked) → **12.3 tok/s**. Even slower than 4x chunked (13.2) because single topk on 128K vocab falls back to single-core `ttnn.sort` (not multi-core `ttnn.topk` which requires power-of-2 < 65536). This is just the old sort bottleneck, not a new finding. The 4x chunked result (13.2) remains the key datapoint.
- [x] **Experiment A3**: `logits.max(dim=-1)` in non-greedy sampler → **19.1 tok/s**. Simple reductions are fast. The slowdown is **specific to topk/sort ops**, not any operation on the 128K vocab. The 4x chunked `ttnn.topk` itself is the bottleneck — it's slow when dispatched from the vLLM sampler program but fast standalone (0.99ms).
- [x] **Experiment A4**: Single multi-core topk on 32K chunk → **18.5 tok/s**. A single `ttnn.topk` is essentially free. The slowdown comes from the **4x chunking loop** (split+pad+4×topk+concat = ~30 extra ops). Each op adds ~0.5ms dispatch overhead → ~15ms total.
- [x] **Experiment A5**: 2x chunks of 65536 → **11.1 tok/s**. Worse — 65536 falls back to single-core `ttnn.sort`. Confirms 4x 32K is optimal given `ttnn.topk` multi-core limit (power-of-2, < 65536).

**Complete experiment results table:**

| Experiment | tok/s | Ops in sampler | Finding |
|---|---|---|---|
| Greedy argmax | 19.0 | 3 | baseline |
| Non-greedy argmax (TT_NONGREEDY_ARGMAX_ONLY) | 18.6 | 3 | non-greedy path is fast |
| Non-greedy max (TT_NONGREEDY_MAX_ONLY) | 19.1 | ~5 | simple reductions are free |
| 1x topk 32K (TT_NONGREEDY_ONE_CHUNK_TOPK) | 18.5 | ~8 | single ttnn.topk is free |
| **4x topk 4×32K (TT_NONGREEDY_TOPK_ONLY)** | **13.2** | **~30** | **4x loop is the bottleneck** |
| 2x topk 2×64K (TT_NONGREEDY_2CHUNK_TOPK) | 11.1 | ~15 | sort fallback, worse |
| Full sampling (TT_USE_TTNN_SAMPLING) | 13.4 | 58 | pad+sampling add nothing over topk |
| Full + penalties | 12.7 | 58+ | penalties add 0.7 tok/s |

**Conclusion: the 4x chunked topk loop creates ~30 compiled ops with ~0.5ms dispatch overhead each in the vLLM sampler context. This is the fundamental bottleneck.**

### ROOT CAUSE FOUND: extra input bindings inflate dispatch overhead (Apr 15)

Standalone 4x topk with vs without metadata input tensors:
- **Without metadata inputs: 1.17 ms/call**
- **With 4 metadata inputs (temp, top_k, top_p, min_p): 6.89 ms/call**

The extra input bindings cause a **6x slowdown** even though the inputs aren't consumed by the topk compute. The XLA/tt-mlir compiler generates a worse program when there are more input tensor bindings.

In the vLLM sampler, the compiled program has ~10 input bindings (logits + temperature + top_k + top_p + min_p + other metadata). This inflates the dispatch overhead for every op in the program, turning the 4x topk loop from 1ms to ~25ms.

**Update:** Hardcoding all params (TT_HARDCODE_SAMPLING_PARAMS) → 13.7 tok/s, same as 13.8. Input bindings are NOT the primary cause in the real model. The standalone test difference (1.17→6.89ms) was misleading — in the full model with ~2000 forward ops, the input binding effect is negligible.

**INPUT BINDING THEORY: DISPROVED by IR comparison.**

| Version | Graph inputs | Graph ops | tok/s |
|---|---|---|---|
| Current (None for top_k/top_p) | 9 | 72 | 13.8 |
| Hardcoded all params | 6 | 63 | 13.7 |
| Greedy (argmax only) | 1 | 3 | 19.0 |

Reducing inputs 9→6 gave no improvement. The standalone test (1.17→6.89ms) was misleading — in small graphs, input binding overhead is proportionally larger, but in the vLLM context with model forward ops, it's negligible.

**MAJOR CORRECTION (Apr 15 tracy with --max-output-tokens 3):**

Steady-state per-op times (OPT-125M, post-warmup):
- TopKOp: **0.077 ms/call** (multi-core confirmed)
- SamplingOp: **0.274 ms/call**
- ConcatOp: **0.042 ms/call**

Previous tracy showing 34ms/call SamplingOp was JIT compilation contamination (first call = 237ms, steady-state = 0.27ms). The sampler ops are NOT the bottleneck in steady-state.

The 13.4 → 19.0 tok/s gap is NOT from sampler op dispatch overhead. The ops are fast. Investigation continues — the gap may be from the model forward graph being compiled/optimized differently when the non-greedy sampler path exists. Each has per-op overhead in the vLLM runtime (~0.5ms each). Single topk = 18.5 tok/s (1 topk op), 4x topk = 13.2 (4 topk + pads + adds + concats). The overhead is from the **number of ops** dispatched as part of the sampler program, not from input bindings.

**Remaining fix options:**
1. Fuse the 4x topk loop into a single composite tt-mlir op (1 dispatch instead of ~30)
2. Extend ttnn.topk to support ≥ 65536 (enables 2x chunks, halving ops)
3. Move topk into the model forward torch.compile scope (ops amortized in larger program)
4. Implement a custom StableHLO-level topk that handles 128K vocab directly

Fix options (ranked by feasibility):
1. **Fuse the 4x topk loop into a single tt-mlir composite op** — dispatches once instead of ~30 times. Requires tt-mlir work to define a composite that does split+pad+4×topk+concat internally.
2. **Extend `ttnn.topk` to support ≥ 65536 inputs** — enables 2x chunks, halving the loop. Requires tt-metal kernel change.
3. **Move topk into the model forward graph** — compile topk as part of the model forward (single large program) instead of the sampler (separate small program). May reduce per-op dispatch overhead since the ops share the same program context.
4. **Reduce per-op dispatch overhead** — general compiler/runtime improvement to make small op dispatches faster.
- [ ] **Experiment B**: Investigate the 25x amplification — why does topk cost 0.99ms standalone but ~25ms in the vLLM sampler compiled program? Possible causes: (a) per-op dispatch overhead scales with total program state, (b) the topk on logits from the model forward has DRAM contention, (c) the sampler compiled program has inherent dispatch latency.

**Critical correction:** The earlier `TT_TOPK_BEFORE_ARGMAX` (19.1 tok/s) and `TT_GREEDY_WITH_SAMPLING_OPS` (18.7 tok/s) experiments were both **testing dead code**. When `all_greedy=True`, `model_runner.sample_from_logits` returns `torch.argmax` at line 2195 — the sampler is never called, so any topk/sampling ops in the sampler are never executed. These experiments only proved argmax is fast, not that topk is free.
- [ ] **Experiment C**: Tracy comparison of sampler program dispatch — measure actual per-op timing in full model context vs standalone to see if dispatch overhead is truly higher.
- [ ] **Experiment D**: Count actual XLA program dispatches — verify that 58 TTNN ops = 58 device dispatches (some may be fused by the compiler into fewer dispatches).

Possible fix approaches:
1. Move padding to runtime (fewer compiled ops)
2. Pre-allocate padded tensors to avoid per-step pad ops
3. Reduce typecasts by ensuring input types match kernel requirements from the start
4. Use a single fused composite op that does topk+pad+sampling in one dispatch

### Next debug steps

The gap is NOT from sampling ops (proven by TT_GREEDY_WITH_SAMPLING_OPS experiment). Focus areas:

1. **Investigate model_runner.py `all_greedy` vs `all_random` paths** — when `all_greedy=False`, the model runner generates and transfers extra sampling metadata tensors (top_k, top_p, temperature, q_samples) to device every decode step. This host-side overhead is invisible to device-only profiling.
2. **Check if non-greedy triggers different graph compilation** — the sampler is compiled as part of the model forward graph via `torch.compile`. Greedy compiles a different graph than non-greedy. Even if the sampling ops are fast, the different graph structure may cause the compiler to make different optimization decisions for the model forward ops.
3. **Tracy comparison** — run tracy on both greedy and non-greedy to compare host-side timing (tensor creation, metadata building, program dispatch) rather than just device ops.
4. **Rebase tt-mlir** — greedy baseline dropped from 20.4 to 19.0 with our tt-mlir branch. Rebasing may recover this.

### Reproducing experiments

All experiments are gated by env vars. Commands to reproduce:

```bash
# Greedy baseline
pytest -svv "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b1-greedy-device]"

# Non-greedy with ttnn.sampling (with penalties)
TT_USE_TTNN_SAMPLING=1 pytest -svv "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b1-nongreedy-device]"

# Non-greedy without penalties (isolates penalty overhead)
TT_USE_TTNN_SAMPLING=1 pytest -svv "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b1-nongreedy-nopenalty-device]"

# Greedy + sampling ops (proves ops don't degrade perf)
TT_GREEDY_WITH_SAMPLING_OPS=1 TT_USE_TTNN_SAMPLING=1 pytest -svv "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b1-greedy-device]"

# Standalone op overhead breakdown
TT_USE_TTNN_SAMPLING=1 python3 perf_debug/test_sampling_op_overhead.py all

# Individual standalone tests
TT_USE_TTNN_SAMPLING=1 python3 perf_debug/test_sampling_op_overhead.py standalone
TT_USE_TTNN_SAMPLING=1 python3 perf_debug/test_sampling_op_overhead.py greedy
TT_USE_TTNN_SAMPLING=1 python3 perf_debug/test_sampling_op_overhead.py topk_only
TT_USE_TTNN_SAMPLING=1 python3 perf_debug/test_sampling_op_overhead.py topk_pad
TT_USE_TTNN_SAMPLING=1 python3 perf_debug/test_sampling_op_overhead.py sampling

# Full overhead breakdown with logs
bash perf_debug/run_sampling_overhead.sh
```

## Current State Summary (Apr 15, updated)

### What was built
- Full tt-mlir pipeline for `ttnn.sampling` op (21 files, branch `kmabee/apr12_vllm_demo_sampling_op_integration`)
- tt-xla custom op registration, sampler integration behind `TT_USE_TTNN_SAMPLING=1`
- Correctness fixed (temperature bug: kernel expects 1/temperature, not raw temperature)
- Output verified coherent on OPT-125M and Llama-3.1-8B

### Performance results (Llama-3.1-8B batch=1)

| Config | tok/s | Env vars |
|---|---|---|
| Greedy baseline (device) | 19.0 | (none) |
| Greedy baseline (CPU) | 15.9 | cpu_sampling=True |
| Non-greedy argmax only | 18.6 | TT_NONGREEDY_ARGMAX_ONLY=1 |
| Non-greedy max only | 19.1 | TT_NONGREEDY_MAX_ONLY=1 |
| 1x topk (32K chunk) | 18.5 | TT_NONGREEDY_ONE_CHUNK_TOPK=1 |
| **4x topk (4×32K chunks)** | **13.2** | TT_NONGREEDY_TOPK_ONLY=1 |
| 2x topk (2×64K, sort fallback) | 11.1 | TT_NONGREEDY_2CHUNK_TOPK=1 |
| Full ttnn.sampling (no penalties) | 13.4/13.8 | TT_USE_TTNN_SAMPLING=1 |
| Full + penalties | 12.7 | TT_USE_TTNN_SAMPLING=1 (temp=0.6, rep=1.1) |
| Full + penalties + trace | 13.8 | TT_USE_TTNN_SAMPLING=1, enable_trace=True |
| **CPU non-greedy sampling** | **14.3** | cpu_sampling=True |
| Split topk (2 programs) | 12.1 | TT_SPLIT_TOPK=1 TT_USE_TTNN_SAMPLING=1 |
| bf16 fast path | 13.1 | TT_BF16_SAMPLING=1 TT_USE_TTNN_SAMPLING=1 |

### ROOT CAUSE CONFIRMED: Per-op dispatch overhead (Apr 15)

**The sampler compiled graph has 50 dispatched TTNN ops.** Each op incurs ~0.5ms host-side dispatch overhead. Total: ~25ms/token, explaining the full gap from greedy (3 ops, ~1.5ms) to non-greedy (50 ops, ~25ms).

IR analysis of the non-greedy sampler graph (graph 9):
| Op type | Count | Notes |
|---|---|---|
| typecast | 17 | Mostly bf16↔f32↔ui16 for topk I/O |
| pad | 6 | Vocab alignment + batch-1→32 for sampling |
| slice_static | 5 | Chunk extraction from padded logits |
| full | 5 | Constant tensors (offsets, epsilon, 1.0) |
| topk | 4 | Multi-core, 32K chunk each |
| to_layout | 3 | Layout conversions for sampling inputs |
| add | 3 | Index offset per chunk |
| concat | 2 | Merge values + indices |
| where | 1 | Greedy/random merge |
| sampling | 1 | ttnn.sampling kernel |
| reshape | 1 | |
| gt | 1 | Temperature comparison |
| divide | 1 | 1/temperature |
| **Total** | **50** | **~25ms at ~0.5ms/op** |

**Device-side execution is fast** (all ops sub-ms per tracy). The overhead is entirely **host-side dispatch latency** — the C++ runtime dispatching each TTNN op through the tt-metal API. Tracy only measures device-side execution, which is why earlier profiling didn't show the overhead.

### Approaches tested and results (Apr 15)

**1. Split topk into separate compiled program (TT_SPLIT_TOPK=1): 12.1 tok/s — WORSE**
- Hypothesis: smaller programs would have less per-op overhead.
- Result: the extra inter-program dispatch (~5ms) outweighed any savings.
- Each torch.compile program has its own dispatch overhead.

**2. Batched topk via reshape (TT_BATCHED_TOPK=1): 15.0 tok/s — broken output**
- Reshape [1, 131072] → [4, 32768] + single topk call.
- Compiled correctly in isolation (verified). But produced garbage (`!!!!!`) in the full sampler graph.
- IR analysis: the compiler decomposed the reshape+topk back into 4 separate slice+topk calls — same ops as the loop.
- The 15.0 tok/s was an artifact of broken computation (some ops short-circuited).

**3. bf16 fast path (TT_BF16_SAMPLING=1): 13.1 tok/s — no improvement**
- Skip `logits.to(float32)` to eliminate bf16→f32→bf16 round-trips.
- The compiler still inserts typecasts (topk requires bf16 input, outputs uint16).
- Net effect: same number of ops, slightly worse precision.

**4. CPU sampling (cpu_sampling=True): 14.3 tok/s — better than compiled!**
- `logits.cpu()` + CPU topk/multinomial.
- Faster than 50-op compiled path because CPU has negligible per-op overhead.
- Gap to greedy (15.9 CPU greedy): ~7ms from CPU sampling compute.
- Gap to device greedy (19.0): ~10ms from XLA sync + data transfer.

**5. Eager TTNN sampling (TT_EAGER_SAMPLING=1): CRASHED**
- Attempted to call ttnn.topk + ttnn.sampling as eager Python ops.
- `ttnn.open_device(0)` conflicts with PJRT-managed device.
- `ttnn.GetDefaultDevice()` returns None (PJRT doesn't set TTNN default).
- Cannot share device between PJRT and TTNN Python bindings.

### Key insight: why CPU is faster than device for sampling

The CPU sampling path (14.3 tok/s) beats the compiled device path (13.4 tok/s) because:
- CPU has negligible per-op overhead (function calls are ~100ns)
- Device has ~0.5ms per-op dispatch overhead (host→device→host round-trip)
- 50 ops × 0.5ms = 25ms device overhead
- CPU topk + multinomial on 128K vocab = ~7ms total
- 7ms < 25ms, so CPU wins

But CPU still loses to device greedy (19.0 vs 15.9 = 3.1 tok/s gap) because `logits.cpu()` forces an XLA sync that breaks pipeline overlap between the model forward and sampling.

### Why this can't be fixed at the Python/sampler level

**The root cause is in the tt-xla runtime, not the sampler code.** The runtime dispatches each TTNN op in a compiled program sequentially via the tt-metal C++ API. Each dispatch involves:
1. Host prepares op arguments
2. Host sends command to device
3. Device executes (fast, sub-ms)
4. Host waits for completion
5. Host prepares next op

Steps 1-2 and 4-5 take ~0.3-0.5ms per op. With 50 ops, this accumulates to ~25ms.

No amount of Python restructuring can change this. The compiler generates 50 TTNN ops because:
- ttnn.topk requires bf16 input (4 typecasts from f32)
- ttnn.topk outputs uint16 indices (4 typecasts to int32)
- ttnn.topk values come as bf16 (4 typecasts to f32 for downstream ops)
- ttnn.sampling requires batch=32 padding (6 pad ops)
- 4 chunk splits, 4 topk calls, 3 offset adds, 2 concats

### What WILL fix it: fused composite ops in tt-mlir

The fix requires tt-mlir changes to reduce the number of dispatched TTNN ops from ~50 to ~5:

**Option A: `tt::topk_sample` fused custom op** (recommended, highest impact)
- Register new custom op that takes (logits, temperature, top_k, top_p)
- Runtime does internally: split vocab → 4x ttnn.topk → merge → ttnn.sampling
- Result: **1 dispatch** instead of ~50. Expected: ~18+ tok/s.
- Implementation: ~20 files across tt-mlir (similar to ttnn.sampling integration)

**Option B: Enhanced topk composite for large vocabs**
- When torch.topk is called with input dim > 32768, the tt-mlir composite lowering should internally split+chunk+merge.
- Result: 1 topk dispatch instead of 4x (slice + typecast + topk + typecast) = saves ~20 ops.
- This benefits ALL topk usage, not just sampling.

**Option C: Reduce per-op dispatch overhead in tt-xla runtime**
- Investigate asynchronous dispatch, op batching, or reducing host-side setup per op.
- Most general fix but most complex. Affects all compiled programs.

**Option D: Extend ttnn.topk to support > 32768 input** — INVESTIGATED, NOT VIABLE as host-side chunking.
- Implemented Approach C (chunked topk with host-side merge) in tt-metal submodule. Correctness passes for 65536, 98304, 100000, 128256 dims.
- Performance: 49ms → 37ms for 128K (25% better than sort), but still 7.5x slower than 5ms baseline for 32K.
- **Bottleneck: host round-trips.** cpu(), to_vector(), HostBuffer construction, to_layout(TILE), to_device() per chunk dominate. Device-side multi-core topk on each 32K chunk is fast; the host glue is slow.
- C++ chunked path (37ms) is SLOWER than existing Python-level 4x chunking (~15ms from compiled ops). Host data transfer overhead > compiled op dispatch overhead.
- **What would work:** on-device two-level merge in a single program factory — chunk+merge entirely on-device without host round-trips. Requires modifying `topk_multi_core_program_factory` to orchestrate a two-level bitonic merge in a single kernel launch. Significantly more invasive but the only path to ~5ms for 128K.
- **Useful side-finding:** TopKCoreConfig uint16→uint32 fix prevents silent overflow for dims >65535. Worth submitting as standalone bug fix to tt-metal.

### `tt::topk_sample` fused custom op — IMPLEMENTED, DEVICE STATE CORRUPTION BLOCKER (Apr 15-16)

Full implementation complete across ~20 files (same pattern as tt::sampling). Builds cleanly. The compiled IR is correct and minimal (3 ops: 2 to_layout + 1 topk_sample). The runtime does 4x chunked topk + merge internally.

**Standalone test: PASSES.** 20ms steady-state, correct global vocab indices, repeated calls work.
```
TT_TOPK_SAMPLE=1 python3 perf_debug/test_topk_sample_compile.py
→ First call: 0.51s, Steady-state: 20.0ms, PASSED
```

**vLLM integration: HANGS during decode.** The runtime's ~25 internal TTNN op allocations corrupt the DRAM address space that the program executor's pre-allocated tensor buffers expect. When the program executor continues to subsequent compiled ops after our runtime returns, it reads/writes to corrupted DRAM.

**Debugging timeline:**
1. Initial run: hang appeared to be in compilation → isolated to runtime (stack trace showed `TransferFromDevice` block)
2. Standalone compile test: passes → confirmed compiler is fine
3. `ttnn::sampling` from within runtime: hangs → split to return only topk indices
4. Phase-by-phase isolation: Phase 1 (topk) works, Phase 2 (merge) works, Phase 3 (sampling) hangs
5. Skipped sampling, returned dummy: works → confirmed sampling kernel interaction
6. Removed sampling from runtime, let compiled graph handle it: precompilation passes, decode hangs
7. Added explicit `ttnn::deallocate` on all intermediates: partially helped (precompilation passed further) but decode still hangs
8. `TT_RUNTIME_SYNC_AFTER_OP=1`: still hangs → NOT an async dispatch issue
9. Skipped logprobs precompilation: precompilation completes, first decode step hangs

**Root cause:** Calling ~25 TTNN device ops (slice, pad, topk, typecast, add, concat, gather) from within a single runtime dispatch allocates intermediate device tensors that corrupt the DRAM layout expected by the program executor's compiled tensor pool. The TTNN APIs manage their own memory allocations, which conflict with the program executor's pre-allocated output buffers. Even with explicit `ttnn::deallocate` on all intermediates and device `Synchronize`, the DRAM allocator state is not restored to what the program executor expects.

**This is a fundamental architectural limitation:** the tt-mlir runtime doesn't support calling raw TTNN APIs from within a custom op dispatch because the TTNN allocator and the program executor's tensor pool use the same DRAM address space without coordination.

**Possible fixes:**
- Implement the fused topk as a **tt-metal program factory** (single kernel launch, no intermediate TTNN API calls) — avoids the allocator conflict entirely
- Add runtime support for **sub-allocations** within custom op dispatches — the program executor would reserve a DRAM region for the custom op's temporaries
- Reduce per-op dispatch overhead (Option 1 in the issue update) — makes the original 50-op approach fast enough

### Recommended path forward (updated Apr 16)

**Priority 1: Reduce per-op dispatch overhead in the program executor.** This is the root cause fix. Each of the 50 compiled TTNN ops costs ~0.5ms in host-side dispatch — far more than GPU kernel launches (~10μs). Profile the dispatch path to identify what's in that 0.5ms (flatbuffer deserialization? tensor pool lookups? L1 allocation? synchronization?). If reduced to ~0.1ms, the existing 50-op sampler would add ~5ms instead of ~25ms → ~17+ tok/s. Benefits ALL compiled programs, not just sampling. Keeps the sampler as pure Python torch ops.

**Priority 2: On-device two-level topk merge in tt-metal.** A single program factory that does chunk+merge entirely on-device without host round-trips or intermediate TTNN API calls. The only path to ~5ms topk on 128K vocab. Requires modifying `topk_multi_core_program_factory` to orchestrate a two-level bitonic merge in a single kernel launch. Significantly more invasive but avoids both the dispatch overhead and the DRAM allocator corruption.

**The `tt::topk_sample` fused custom op approach (calling TTNN APIs from runtime) is NOT viable** with the current tt-mlir runtime architecture. The implementation is complete and correct but blocked by DRAM allocator conflicts between the custom op's internal TTNN calls and the program executor's tensor pool.

### Commands for benchmarking

```bash
# Greedy baseline (device)
pytest -svv "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b1-greedy-device]"

# Greedy baseline (CPU)
pytest -svv "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b1-greedy-cpu]"

# Non-greedy with ttnn.sampling (no penalties)
TT_USE_TTNN_SAMPLING=1 pytest -svv "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b1-nongreedy-nopenalty-device]"

# Non-greedy CPU sampling
pytest -svv "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b1-nongreedy-cpu]"

# Isolation experiments (use with nongreedy-nopenalty-device config):
TT_NONGREEDY_ARGMAX_ONLY=1 TT_USE_TTNN_SAMPLING=1 pytest -svv ...  # argmax only
TT_NONGREEDY_MAX_ONLY=1 TT_USE_TTNN_SAMPLING=1 pytest -svv ...     # max reduction
TT_NONGREEDY_ONE_CHUNK_TOPK=1 TT_USE_TTNN_SAMPLING=1 pytest ...    # 1x topk
TT_NONGREEDY_TOPK_ONLY=1 TT_USE_TTNN_SAMPLING=1 pytest ...         # 4x topk only

# Split topk (DISPROVED — slower due to inter-program overhead)
TT_SPLIT_TOPK=1 TT_USE_TTNN_SAMPLING=1 pytest -svv ...

# bf16 fast path (DISPROVED — compiler still inserts typecasts)
TT_BF16_SAMPLING=1 TT_USE_TTNN_SAMPLING=1 pytest -svv ...

# Standalone overhead tests
TT_USE_TTNN_SAMPLING=1 python3 perf_debug/test_sampling_op_overhead.py all
bash perf_debug/run_sampling_overhead.sh

# IR dump and extraction
TTXLA_LOGGER_LEVEL=DEBUG TT_USE_TTNN_SAMPLING=1 pytest -svv --max-output-tokens 3 "..." &> /tmp/ir_dump.log
python3 /localdev/kmabee/scripts/extract_mlir_graphs.py --subdir /tmp/ir_output --type ttnn /tmp/ir_dump.log
```

## Implementation: Runtime Chunked TopK (Option B)

The highest-impact fix is to add vocab-chunking logic inside the tt-mlir topk runtime. When `torch.topk(logits, k=32)` is called on [1, 128256], the compiler generates a single `ttnn.topk` op. The runtime detects the large dimension and internally splits into multi-core-friendly chunks.

### Changes required

**1. tt-mlir runtime** (`runtime/lib/ttnn/operations/reduction/topk.cpp`):

Add `chunkedTopk()` that:
- Splits input along last dim into chunks of ≤ 32768
- Pads each chunk to next power-of-2
- Calls `ttnn::topk(chunk, k)` on each
- Adds per-chunk index offsets
- Concatenates results → `[batch, numChunks * k]`
- Runs final `ttnn::topk(merged, k)` to reduce back to `[batch, k]`
- Uses `ttnn::gather` to map local indices back to global vocab indices

The `run()` function checks `shouldChunkTopk()` (last dim > 32768) and dispatches to `chunkedTopk()` instead of the direct `::ttnn::topk()` call.

Key APIs: `ttnn::slice`, `ttnn::pad`, `ttnn::topk`, `ttnn::typecast`, `ttnn::concat`, `ttnn::gather`, `ttnn::full`, `ttnn::add`.

**2. tt-xla sampler** (`integrations/vllm_plugin/vllm_tt/sampler.py`):

Replace `apply_top_k_top_p_fast()` (4x Python loop) with:
```python
vals, inds = torch.topk(logits, k=_TOPK_K_PER_CHUNK, dim=-1)
```

This compiles to 1 TTNN topk op instead of ~20 ops (4 slices + 4 pads + 4 topk + 4 adds + 2 concats + typecasts).

**3. Estimated improvement**:
- Current sampler: ~50 ops, ~25ms overhead → 13.4 tok/s
- With runtime chunking: ~10 ops (1 topk + typecasts + sampling), ~5ms overhead → ~17-18 tok/s
- Target: 19.0 tok/s (greedy)

**4. Build/test**: requires tt-mlir rebuild. The runtime change is in the tt-mlir submodule (branch `kmabee/apr12_vllm_demo_sampling_op_integration`). After committing the runtime change, update `TT_MLIR_VERSION` hash in `third_party/CMakeLists.txt`.

**Already prepared**: `TT_RUNTIME_TOPK=1` env var in `sampler.py` enables the single-topk path for testing once the runtime is built.

## Branches

**tt-xla:** `kmabee/vllm_perf_apr12` based on `866d0abdd` (Apr 12, 2026)

**tt-mlir (SamplingOp):** `kmabee/apr12_vllm_demo_sampling_op_integration` at `7c9c730bb`
- Full ttnn.sampling integration (21 files, working in production)

**tt-mlir (TopKSampleOp):** `kmabee/apr12_vllm_demo_sampling_op_integration_TopKSampleOp_integration` at `4bd11c845`
- Full TopKSampleOp integration (18 files on top of SamplingOp branch)
- Works standalone (20ms), blocked in vLLM by DRAM allocator conflict
- Preserved as reference for future runtime architecture improvements
