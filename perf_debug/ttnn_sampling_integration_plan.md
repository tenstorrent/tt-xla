# Plan: Integrate ttnn.sampling() Fused Custom Op

Tracking issue: https://github.com/tenstorrent/tt-xla/issues/3940

## Overview

Replace the 66-op compiled sampling graph with `ttnn.sampling()` — a native fused kernel in tt-metal that does top-k, top-p, temperature scaling, softmax, and multinomial sampling in a single device op. Follow the same integration pattern as paged attention (`tt::paged_update_cache`).

Key constraint: `ttnn.sampling()` requires batch dim = 32 (hardcoded). Batch<32 needs padding.

## Phase 1: Verify the op on hardware

1. **Run the existing tt-metal unit test**:
   - `third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tests/ttnn/unit_tests/operations/eltwise/test_sampling.py`
   - If it fails, stop — the kernel doesn't work on our hardware.

2. **Write a standalone test** (`perf_debug/test_ttnn_sampling_direct.py`):
   - Llama vocab_size=128256, batch=1 padded to 32
   - Verify output shape [1,1,1,32], token indices in [0, vocab_size)
   - Test edge cases: temp=0 (greedy), top_k=1, top_p=1.0
   - Test determinism with same seed
   - Test that k/p/temp can vary per-user in the batch

### Phase 1 results (2026-03-26)

**Critical finding: `ttnn.sampling()` overflows L1 at vocab >= 8192.** Circular buffers need ~33MB at Llama vocab (128256) but only ~1.5MB L1 per core. The existing unit test only uses tiny vocab sizes (64, 256).

**Production models never feed full vocab to `ttnn.sampling()`.** Reading `models/common/sampling/tt_sampling.py` revealed the actual pipeline:
1. Split logits in half along vocab dim
2. `ttnn.topk()` each half → `max_top_k` (typically 32) results per half
3. Concat → `2 * max_top_k` candidates (64 tokens)
4. Add per-half device offsets to get global vocab indices
5. `ttnn.sampling()` on the reduced set (64 tokens, well within L1)

This means **the custom op must wrap topk + sampling, not just sampling alone.**

All correctness tests pass with the topk+sampling pipeline:
- Small vocab direct sampling (4096): PASS
- Llama vocab (128256) via topk+sampling: PASS
- OPT vocab (50272) via topk+sampling: PASS
- Determinism (same seed): PASS
- Greedy (top_k=1): PASS — 32/32 match argmax
- Per-user k/p/temp: PASS
- top_p=1.0 (disabled): PASS

Answers to open questions:
- **temp=0 greedy**: Not tested directly, but top_k=1 with p=0.0 gives exact argmax (32/32 match)
- **Per-request seeds**: `ttnn.sampling()` takes a single global seed, but production uses `ttnn.manual_seed()` with per-user seeds before calling it
- **top_k=0 (disabled)**: Not tested — production uses max_top_k=32 as the upper bound
- **Per-user params**: Confirmed working — k/p/temp can all vary per user

## Phase 2: Performance comparison

1. Benchmark `ttnn.sampling()` over 100 iterations (after warmup) at Llama vocab size
2. Record mean/min/max latency
3. Compare against the known 93ms/token baseline from the 66-op graph

### Phase 2 results (2026-03-26)

| Pipeline | Vocab | P50 Latency | Baseline (66-op) | Speedup |
|---|---|---|---|---|
| sampling only (reduced vocab=64) | 64 | 0.18ms | — | — |
| topk + sampling | 50,272 (OPT) | **10.7ms** | ~93ms | **~8.7x** |
| topk + sampling | 128,256 (Llama) | **46.0ms** | ~147ms | **~3.2x** |

The sampling op itself is sub-millisecond. The bottleneck is `ttnn.topk()` on large vocabs. Note: these measurements include host-side tensor creation overhead (creating ttnn tensors each iteration). In production with persistent device tensors, the actual improvement will be larger.

## Phase 3: Custom op registration in tt-xla

**File**: `python_package/tt_torch/custom_ops.py`

Follow the `paged_update_cache` pattern exactly (lines 452-503).

**Updated after Phase 1**: The custom op must accept pre-topk'd inputs (reduced vocab, ~64 tokens after the split-topk-concat pipeline), NOT the full vocab. The topk reduction happens in the sampler before calling this op, or we create a higher-level op that wraps topk+sampling together.

Two options being considered:
- **Option A**: Custom op wraps only `ttnn.sampling()` (reduced inputs). The topk pipeline is compiled normally via `torch.compile` (torch.topk -> ttnn.topk lowering already exists).
- **Option B**: Custom op wraps the full topk+sampling pipeline. Avoids compiling topk separately.

1. `@torch.library.custom_op("tt::sampling", mutates_args=[], device_types=["xla", "cpu"])`
   - Inputs: `input_values` [1,1,32,reduced_vocab], `input_indices` [1,1,32,reduced_vocab], `k` [1,1,32], `p` [1,1,32], `temp` [1,1,32], `seed: int`
   - XLA path: `stablehlo_custom_call` with target `"tt.sampling"`, seed as frontend attribute
   - CPU fallback: reference Python implementation (softmax -> top-k -> top-p -> multinomial)
   - Output: `[1,1,1,32]` int32

2. `@sampling.register_fake` — returns `torch.zeros((1,1,1,32), dtype=torch.int32)`

Note: k/p/temp tensors reshaped to [1,1,32] (3D minimum) for `stablehlo_custom_call` compatibility.

## Phase 4: Compiler lowering in tt-mlir

~8 files across the compiler. Follow `paged_update_cache` pattern everywhere.

### 4a: TTIR op definition
- **File**: `third_party/tt-mlir/.../include/ttmlir/Dialect/TTIR/IR/TTIROps.td`
- Add `TTIR_SamplingOp` following `TTIR_PagedUpdateCacheOp` (line 3222)
- Arguments: input_values, input_indices, k, p, temp tensors + seed attribute

### 4b: TTNN op definition
- **File**: `third_party/tt-mlir/.../include/ttmlir/Dialect/TTNN/IR/TTNNOps.td`
- Add `TTNN_SamplingOp` following `TTNN_PagedUpdateCacheOp` (line 1039)
- NOT an inplace op (produces new output tensor)

### 4c: StableHLO -> TTIR conversion
- **File**: `third_party/tt-mlir/.../lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`
- Add `StableHLOSamplingConversionPattern` matching `funcName == "tt.sampling"` (follow line 6048)
- Parse seed from frontend_attributes, verify 5 operands + 1 result

### 4d: TTIR -> TTNN conversion
- **File**: `third_party/tt-mlir/.../lib/Conversion/TTIRToTTNN/TTIRToTTNN.cpp`
- Add `SamplingOpConversionPattern` (follow line 741)

### 4e: Flatbuffer schema
- **New file**: `.../include/ttmlir/Target/TTNN/operations/sampling.fbs`
- Add `SamplingOp` table with 5 tensor refs + seed attr
- Add to `OpType` union in `program.fbs`

### 4f: Flatbuffer serialization
- **File**: `.../lib/Target/TTNN/TTNNToFlatbuffer.cpp`
- Add `createOp` for SamplingOp (follow line 1632)

### 4g: Runtime dispatch
- **New file**: `.../runtime/lib/ttnn/operations/sampling/sampling.cpp`
- Extract tensors, call `ttnn::sampling(...)` (follow paged_update_cache runtime)
- Add dispatch case in `program_executor.cpp` (line 452)
- Update `CMakeLists.txt` for new operation directory

### 4h: EmitC/EmitPy conversion
- Add patterns in `TTNNToEmitC.cpp` and `TTNNToEmitPy.cpp`

### 4i: MLIR lit test
- **New file**: `.../test/ttmlir/Conversion/StableHLOToTTIR/transformer/sampling.mlir`

## Phase 5: Sampler integration

### sampler.py changes
**File**: `integrations/vllm_plugin/vllm_tt/sampler.py`

Current `Sampler.sample()` (lines 140-204) does: masks -> penalties -> greedy argmax -> temp scaling -> min_p -> top-k/top-p -> softmax -> Gumbel-max -> select.

New fused path replaces steps 3-8:
1. Apply masks and penalties (steps 1-2 stay as-is — fused op doesn't handle these)
2. `torch.sort(logits, descending=True)` -> sorted values + indices
3. Build per-user k/p/temp tensors from `XLASupportedSamplingMetadata`
4. Pad batch to 32 if needed
5. Call `torch.ops.tt.sampling(sorted_values, sorted_indices, k, p, temp, seed)`
6. Unpad back to actual batch size

### model_runner.py changes
**File**: `integrations/vllm_plugin/vllm_tt/model_runner.py`

- Add precompilation trace for the fused path in `_precompile_sample_from_logits` (line 1817)
- Branch on config flag in `sample_from_logits` (line 2152)

### Feature gate
`TT_USE_FUSED_SAMPLING=1` env var (default off until validated). Keep old path as fallback.

## Phase 6: End-to-end validation

1. Run existing sampling tests with fused path enabled
2. Greedy decoding: verify identical tokens vs old path
3. Run OPT-125M inference end-to-end
4. Measure tokens/sec improvement

## Dependency graph

```
Phase 1 (verify op) -> Phase 2 (benchmark)
                            |
               Phase 3 (custom_ops.py)  --+
                                          +--> Phase 5 (sampler wiring) --> Phase 6 (validation)
               Phase 4 (tt-mlir)        --+
```

Phases 3 and 4 can proceed in parallel. Phase 5 requires both.

## Open questions to resolve during Phase 1

1. Does `temp=0` degenerate to argmax, or do we need a separate greedy path?
2. Does the op support per-request seeds, or just one global seed?
3. What happens with `top_k=0` (disabled) — does it keep all tokens?
4. Is `ttnn.sort` on vocab_size=128K tensors reliable? (relates to the known sort+scatter corruption issue)
5. What's the precision of `sub_core_grids` — can we omit it and let the kernel choose defaults?

## Key files reference

| File | Role |
|---|---|
| `python_package/tt_torch/custom_ops.py` | Custom op registration (paged_update_cache pattern at line 452) |
| `integrations/vllm_plugin/vllm_tt/sampler.py` | Current 66-op sampling implementation |
| `integrations/vllm_plugin/vllm_tt/model_runner.py` | Decode loop, sample_from_logits at ~line 2152 |
| `integrations/vllm_plugin/vllm_tt/metadata.py` | XLASupportedSamplingMetadata with k/p/temp tensors |
| `third_party/tt-mlir/.../tests/ttnn/unit_tests/operations/eltwise/test_sampling.py` | Existing unit test |
| `third_party/tt-mlir/.../ttnn/cpp/ttnn/operations/reduction/sampling/` | tt-metal kernel source |
