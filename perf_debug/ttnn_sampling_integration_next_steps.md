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

## Branch

`kmabee/vllm_perf_apr12` based on `866d0abdd` (tt-xla Apr 12, 2026)
tt-mlir branch: `kmabee/apr12_vllm_demo_sampling_op_integration` at commit `ffc4810ce`
