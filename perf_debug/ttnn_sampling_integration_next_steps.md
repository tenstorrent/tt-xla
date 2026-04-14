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

### Performance results (functionally passing but output is garbage)
| Config | Without ttnn.sampling | With ttnn.sampling | Greedy |
|---|---|---|---|
| OPT-125M b1 | 9.1 tok/s | 9.3 tok/s | 11.3 tok/s |
| Llama-3.1-8B b1 | 10.9 tok/s | 10.7 tok/s | ~19 tok/s |

### Correctness bug: garbage output
Both OPT-125M and Llama-3.1-8B produce incoherent text with `TT_USE_TTNN_SAMPLING=1`:
- OPT-125M: `<s> +D&&<s><s> :1 > Copy files out the process-type-<s> 3 +B and return 1 if 2: COD has left D...`
- Llama-3.1-8B: `Keep Functions! Small & single -! Simple Function! Use One line , a little.\nFunction.\nWrite!! Function...`

Token IDs are in-range (no crashes), but the sampling distribution is wrong.

**Likely root cause: double temperature application.** The sampler applies `logits / temperature` (line 222 in sampler.py) before passing to `_ttnn_sampling_padded`, which then passes the `temp` tensor to `ttnn.sampling`. The kernel applies temperature again internally. This means temperature is applied twice, distorting the distribution.

**Other possible causes:**
1. Top-k value mismatch: the `k` tensor passed to ttnn.sampling is `sampling_metadata.top_k` (the user's requested top-k, e.g. 64), but the candidate set from 4x topk is already 128 tokens. If k < candidate count, ttnn.sampling further filters — but if k > candidate count, the kernel may behave unexpectedly.
2. Top-p value interpretation: the sampler's `top_p` might be in a different range than what the kernel expects.
3. The `seed=42` is hardcoded — might cause deterministic-but-wrong behavior vs the intended stochastic sampling.

### Debugging plan
1. **Test 1: Fix double temperature** — Pass `temp=1.0` (no-op) to ttnn.sampling since temperature is already applied to the logits. If output becomes coherent, this was the root cause.
2. **Test 2: Verify ttnn.sampling standalone** — Feed known logits (e.g., one-hot or sharply peaked) to the custom op and verify the kernel picks the expected token.
3. **Test 3: Compare distributions** — For a single decode step, capture the logits before and after topk, run both the existing Gumbel-max path and ttnn.sampling, compare selected token IDs.

### Op count overhead analysis
The compiled non-greedy graph has 62 non-bookkeeping TTNN ops vs 50 for greedy+topk. The 12 extra ops are: temperature divide on full 128K vocab, 6 padding ops (batch 1→32), torch.where+gt (greedy/random merge), and 6 extra typecasts. The 19 typecasts total (vs 13 in greedy) are the largest single category.

### Performance improvement opportunities
1. **Eliminate batch padding at batch=32** — kernel is designed for batch=32; padding overhead disappears
2. **Move padding into runtime** — reduce compiled graph op count
3. **Skip temperature in sampler** — let ttnn.sampling handle it (also fixes correctness)
4. **Pass top-k/top-p as attributes instead of tensors** — if all requests share the same values

## Branch

`kmabee/vllm_perf_apr12` based on `866d0abdd` (tt-xla Apr 12, 2026)
tt-mlir branch: `kmabee/apr12_vllm_demo_sampling_op_integration` at commit `ffc4810ce`
