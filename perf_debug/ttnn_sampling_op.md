# Using ttnn.sampling() to Replace the Compiled Sampling Graph

Tracking issue: https://github.com/tenstorrent/tt-xla/issues/3940

## Problem

The current non-greedy sampling path compiles 66 separate TTNN ops into a graph via
`torch.compile(backend="tt")`. This results in:
- 94ms of device FW time per token (vs 0.8ms for greedy argmax)
- Accumulation (cumsum) running on 1 core out of 110
- FillPad running on 1 core out of 110
- 0% FPU utilization — all ops are data-movement bound
- A 66-op dispatch sequence with latency between each op

## Proposed Solution

`ttnn.sampling()` is a native fused kernel in tt-metal that performs top-k, top-p,
temperature scaling, and multinomial sampling in a **single device op**. It already
exists, is Python-bound, and is used in production in other tt-metal model demos
(DeepSeek, Llama3-70B, Qwen3).

This is the same pattern as paged attention (`SdpaDecodeDeviceOperation`), which is a
custom TTNN op exposed via `torch.ops.tt.paged_update_cache()` and called directly from
the vLLM integration instead of being compiled from Python.

## ttnn.sampling() Signature

```python
ttnn.sampling(
    input_values_tensor,   # logits: [1, 1, 32, vocab_size] BFLOAT16, TILE layout
    input_indices_tensor,  # token indices: [1, 1, 32, vocab_size] UINT32/INT32, ROW_MAJOR
    k,                     # top-k per user: [32] UINT32, ROW_MAJOR
    p,                     # top-p per user: [32] BFLOAT16, ROW_MAJOR
    temp,                  # temperature per user: [32] BFLOAT16, ROW_MAJOR
    seed=None,             # optional: int for deterministic sampling
    sub_core_grids=None,   # optional: CoreRangeSet for multicore placement
    output_tensor=None,    # optional: preallocated output
)
# Returns: [1, 1, 1, 32] UINT32 — one sampled token index per user
```

What it does internally (per the docstring in `sampling_nanobind.cpp`):
1. Applies softmax to convert logits to probabilities
2. Applies top-k filter (keeps k highest-probability values)
3. Applies top-p filter (cumulative probability ≤ p)
4. Performs multinomial sampling (random threshold against cumulative probs)
5. Returns `input_indices_tensor[final_index]`

## Key Constraint: Batch Size = 32

The op hardcodes `N * C * H == 32` (from `sampling_device_operation.cpp:47`):
```cpp
TT_FATAL(input_shape[0] * input_shape[1] * input_shape[2] == 32, "Input must have 32 users!");
```

For batch=1, we pad to 32 users. For batch=32, it maps directly. This is the same
approach used in tt-metal model demos. Inner dim (vocab_size) must be divisible by 32.

OPT-125M vocab=50272: 50272 % 32 = 0 ✓
Llama vocab=128256: 128256 % 32 = 0 ✓

## Quick Test (Direct ttnn Call)

Before requesting compiler integration, test the op directly to validate correctness
and performance. Use the existing tt-metal unit test as a reference:
`third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tests/ttnn/unit_tests/operations/eltwise/test_sampling.py`

Minimal test:
```python
import ttnn
import torch

device = ttnn.open_device(device_id=0)

vocab_size = 128256  # Llama vocab
batch_padded = 32    # must be exactly 32

# Logits: [1, 1, 32, vocab_size] BFLOAT16 TILE DRAM
logits = torch.randn(1, 1, batch_padded, vocab_size, dtype=torch.bfloat16)
values_tt = ttnn.from_torch(
    logits, device=device, dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
)

# Indices: [1, 1, 32, vocab_size] UINT32 ROW_MAJOR DRAM
indices = torch.arange(vocab_size).unsqueeze(0).unsqueeze(0).repeat(1, 1, batch_padded, 1).to(torch.int32)
indices_tt = ttnn.from_torch(
    indices, device=device, dtype=ttnn.uint32,
    layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
)

# k, p, temp: [32] per user
k_tt = ttnn.from_torch(torch.full((32,), 50, dtype=torch.int32), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
p_tt = ttnn.from_torch(torch.full((32,), 0.9), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
temp_tt = ttnn.from_torch(torch.full((32,), 0.8), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

# Sample
output = ttnn.sampling(values_tt, indices_tt, k=k_tt, p=p_tt, temp=temp_tt)
# output shape: [1, 1, 1, 32]
sampled_token = ttnn.to_torch(output)[0, 0, 0, 0]  # token for user 0
print(f"Sampled token: {sampled_token}")

ttnn.close_device(device)
```

Run the existing unit test for full validation:
```bash
# From tt-metal source tree
pytest tests/ttnn/unit_tests/operations/eltwise/test_sampling.py -svv
```

## Integration Path (Paged Attention Pattern)

The paged attention op is the exact template. Three steps are required:

### Step 1: Custom op in `python_package/tt_torch/custom_ops.py`

Add `torch.ops.tt.sampling()` following the `paged_update_cache` pattern:
- `@torch.library.custom_op("tt::sampling", ...)` decorator
- XLA path: calls `stablehlo_custom_call` with `"tt.sampling"` op name
- CPU fallback: runs Python-based sampling for testing without hardware
- Register fake/autograd implementations

### Step 2: Compiler lowering in tt-mlir

Add lowering from `tt.sampling` StableHLO custom call → `ttnn::sampling` in the
tt-mlir compiler. This is the work for the compiler team, following the same pattern
as paged attention lowering.

### Step 3: Sampler call site in `integrations/vllm_plugin/vllm_tt/sampler.py`

Replace the current `Sampler` module (66 ops via torch ops) with a direct call to
`torch.ops.tt.sampling()`. This call:
- Bypasses `torch.compile` entirely (like paged attention does)
- Runs the single fused kernel directly
- Handles batch padding (1 → 32) and unpadding

## Files to Modify

| File | Change | Owner |
|---|---|---|
| `python_package/tt_torch/custom_ops.py` | Add `tt::sampling` custom op with XLA + CPU paths | tt-xla team |
| `integrations/vllm_plugin/vllm_tt/sampler.py` | Call `torch.ops.tt.sampling()` instead of compiled graph | tt-xla team |
| `integrations/vllm_plugin/vllm_tt/metadata.py` | Pass k/p/temp tensors in correct [32] shapes | tt-xla team |
| tt-mlir: add op lowering `tt.sampling` → `ttnn::sampling` | Compiler lowering | compiler team |

## Source Files (tt-metal)

```
ttnn/cpp/ttnn/operations/reduction/sampling/
├── sampling.hpp                          # Public API
├── sampling.cpp                          # Public wrapper calling ttnn::prim::sampling
├── sampling_nanobind.hpp/cpp             # Python bindings (ttnn.sampling)
└── device/
    ├── sampling_device_operation.hpp/cpp # Validation, output spec, tensor creation
    ├── sampling_device_operation_types.hpp
    ├── sampling_program_factory.hpp/cpp  # GPU program, circular buffers, kernels
    └── kernels/
        ├── compute/sampling.cpp          # Compute kernel
        └── dataflow/{reader,writer}*.cpp
```

Existing tests using this op in production models:
- `models/common/tests/test_sampling.py` — Qwen3/T3K
- `models/demos/deepseek_v3/tests/test_sampling.py`
- `models/demos/llama3_70b_galaxy/tests/unit_tests/test_sampling.py`
- `tests/ttnn/unit_tests/operations/eltwise/test_sampling.py` — unit test

## Expected Performance Impact

Current (66-op compiled graph, OPT-125M vocab=50272):
- Device FW: 94ms per token
- Key bottlenecks: Accumulation 24ms (1 core), FillPad 14ms (1 core), Sort 14ms, Softmax 7ms
- FPU utilization: 0% (all data movement)

Expected (ttnn.sampling fused kernel):
- Single op dispatch (vs 66 separate dispatches)
- Accumulation and FillPad parallelized across cores in the fused kernel
- Based on tt-metal model demos running at production scale, should be significantly faster
- Exact number TBD — benchmark with direct ttnn call before integration

## Next Steps

- [ ] Run `test_sampling.py` unit test to verify op works on target hardware
- [ ] Write a standalone perf benchmark calling `ttnn.sampling()` directly to measure throughput vs current 66-op graph
- [ ] If perf is good, request compiler team to add tt-mlir lowering for `tt.sampling`
- [ ] Implement `tt::sampling` custom op in `custom_ops.py`
- [ ] Update `sampler.py` to call `torch.ops.tt.sampling()`
- [ ] Run existing sampling integration tests to validate correctness
