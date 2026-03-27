# Context: Integrating ttnn.sampling() into vLLM on TT-XLA

This document summarizes findings from the sampling performance investigation and provides the starting context for implementing the `ttnn.sampling()` integration.

## Tracking Issue

https://github.com/tenstorrent/tt-xla/issues/3940

## The Problem

Non-greedy sampling on TT device compiles a 66-op TTNN graph via `torch.compile(backend="tt")` that adds **+93ms device FW time per token** vs 0.8ms for greedy argmax. This cuts decode throughput roughly in half:

| Config (OPT-125M, batch=1) | tok/s | Sampling overhead |
|---|---|---|
| Greedy device | 11.6 | baseline |
| Non-greedy CPU | 11.0 | +5ms |
| Non-greedy device | 5.9 | **+82ms** |

At Llama 8B vocab (128K), the device sampling overhead grows to **+147ms/token**.

The bottleneck ops in the compiled graph (from Tracy device profiling):
- `Accumulation` (cumsum): 24ms on **1 core** out of 110
- `FillPad` (ttnn.full x11): 14ms on **1 core**
- `Sort`: 14ms on 110 cores
- `Softmax` (x3): 7ms on 110 cores
- 0% FPU utilization across all sampling ops (all data movement bound)

## The Solution: ttnn.sampling()

`ttnn.sampling()` is a native fused kernel in tt-metal that does top-k, top-p, temperature scaling, softmax, and multinomial sampling in a **single device op**. It's already used in production tt-metal model demos (DeepSeek, Llama3-70B, Qwen3).

### Signature

```python
ttnn.sampling(
    input_values_tensor,   # logits: [1, 1, 32, vocab_size] BFLOAT16, TILE layout
    input_indices_tensor,  # token indices: [1, 1, 32, vocab_size] UINT32/INT32, ROW_MAJOR
    k,                     # top-k per user: [32] UINT32, ROW_MAJOR
    p,                     # top-p per user: [32] BFLOAT16, ROW_MAJOR
    temp,                  # temperature per user: [32] BFLOAT16, ROW_MAJOR
    seed=None,             # optional: int for deterministic sampling
    sub_core_grids=None,   # optional: CoreRangeSet
    output_tensor=None,    # optional: preallocated output
)
# Returns: [1, 1, 1, 32] UINT32 - one sampled token index per user
```

### Hard Constraint: Batch Dim Must Be Exactly 32

From `sampling_device_operation.cpp:47`:
```cpp
TT_FATAL(input_shape[0] * input_shape[1] * input_shape[2] == 32, "Input must have 32 users!");
```

For batch=1 we pad to 32 users and extract user 0's result. Vocab must be divisible by 32 (OPT 50272 and Llama 128256 both pass).

### What It Does Internally

1. Softmax to convert logits to probabilities
2. Top-k filter (keeps k highest values)
3. Top-p filter (cumulative probability <= p)
4. Multinomial sampling (random threshold against cumulative probs)
5. Returns `input_indices_tensor[final_index]`

## Integration Pattern: Follow Paged Attention

The paged attention custom op (`tt::paged_update_cache`) is the exact template. It bypasses `torch.compile` and dispatches a fused TTNN kernel directly.

### Step 1: Custom Op in tt-xla (`python_package/tt_torch/custom_ops.py`)

Add `torch.ops.tt.sampling()` following the `paged_update_cache` pattern (line ~453):
- `@torch.library.custom_op("tt::sampling", ...)` decorator
- XLA path: calls `stablehlo_custom_call` with `"tt.sampling"` op name
- CPU fallback: runs Python-based sampling for testing without hardware
- Register `@sampling.register_fake` for shape inference

### Step 2: Compiler Lowering in tt-mlir (compiler team)

Add lowering from `tt.sampling` StableHLO custom call -> `ttnn::sampling`. Same pattern as paged attention lowering. This is compiler team work.

### Step 3: Call Site in Model Runner

Replace the compiled `sample_from_logits()` in `integrations/vllm_plugin/vllm_tt/model_runner.py` with a direct call to `torch.ops.tt.sampling()`.

Current flow (model_runner.py ~line 2152):
```python
@torch.compile(backend="tt", fullgraph=True, dynamic=False)
def sample_from_logits(self, logits, sampling_metadata):
    return self.sampler(logits, sampling_metadata)
```

The `self.sampler` is `integrations/vllm_plugin/vllm_tt/sampler.py:Sampler`, which implements top-k/top-p/temperature/multinomial as individual torch ops that get compiled into 66 TTNN ops.

New flow would call `torch.ops.tt.sampling()` directly, bypassing `torch.compile` entirely, similar to how paged attention is called.

## Key Files

| File | Role |
|---|---|
| `integrations/vllm_plugin/vllm_tt/model_runner.py` | Main decode loop, `sample_from_logits()` at ~line 2152, `sample_from_logits_cpu()` at ~line 2174, precompilation at ~line 1817 |
| `integrations/vllm_plugin/vllm_tt/sampler.py` | Current `Sampler` module with 66-op torch implementation |
| `integrations/vllm_plugin/vllm_tt/metadata.py` | `XLASupportedSamplingMetadata` - carries temperature, top_p, top_k per request |
| `python_package/tt_torch/custom_ops.py` | Where to add `tt::sampling` custom op (reference: `tt::paged_update_cache` at line ~453) |
| `tests/integrations/vllm_plugin/sampling/test_sampling_perf.py` | Synthetic sampling perf benchmarks (no model, ~34s full suite) |
| `tests/integrations/vllm_plugin/sampling/conftest.py` | Test fixtures for sampling tests |

## tt-metal Source Files for the Op

```
ttnn/cpp/ttnn/operations/reduction/sampling/
├── sampling.hpp/cpp                    # Public API
├── sampling_nanobind.hpp/cpp           # Python bindings
└── device/
    ├── sampling_device_operation.hpp/cpp  # Validation, constraints
    ├── sampling_program_factory.hpp/cpp   # GPU program setup
    └── kernels/compute/sampling.cpp       # Compute kernel
```

Existing tests using this op:
- `tests/ttnn/unit_tests/operations/eltwise/test_sampling.py` - unit test
- `models/common/tests/test_sampling.py` - Qwen3/T3K
- `models/demos/deepseek_v3/tests/test_sampling.py`

## Immediate First Steps

1. **Run the existing tt-metal unit test** to verify the op works on your hardware:
   ```bash
   pytest third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tests/ttnn/unit_tests/operations/eltwise/test_sampling.py -svv
   ```

2. **Write a standalone perf benchmark** calling `ttnn.sampling()` directly at OPT-125M (50K vocab) and Llama (128K vocab) sizes. Compare against the 64ms/144ms synthetic numbers from the compiled graph.

3. **If perf is good**, implement the custom op in `custom_ops.py` and wire it into `model_runner.py`.

## Existing CPU Sampling Fallback

There's already a `cpu_sampling` config flag (`tt_config.cpu_sampling`) that routes to `sample_from_logits_cpu()` which pulls logits to host and samples with PyTorch CPU ops. This adds only +5ms at OPT vocab sizes. The `ttnn.sampling()` integration should be a third path alongside device-compiled and CPU.

## Detailed Investigation Notes

Full profiling data, Tracy captures, op breakdowns, and ASCII timeline diagrams are in:
- `perf_debug/sampling_overhead.md` - comprehensive investigation doc
- `perf_debug/ttnn_sampling_op.md` - detailed op spec and integration plan
- `perf_debug_e2e/tracy/` - Tracy profiling captures with annotated CSVs

## Known Issues to Watch For

1. **XLA sort+scatter corruption** (documented in memory): `sort` returns incorrect indices for large tensors (vocab=50272+). The current workaround forces `forward_tpu` path in the sampler to avoid scatter-back. Since `ttnn.sampling()` does its own internal sort, this may not apply, but verify correctness carefully.

2. **Batch padding**: The op requires exactly 32 users. For batch<32, pad logits and params tensors to 32, run the op, extract only the real users' results.

3. **Input prep**: The op expects pre-sorted `input_indices_tensor` alongside `input_values_tensor`. The indices tensor is just `arange(vocab_size)` repeated for each user - it tells the op which token ID each position maps to.
