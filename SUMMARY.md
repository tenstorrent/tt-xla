loader_path: third_party.tt_forge_models.deepseek_r1_distill_llama_8b_quantized_w4a16.causal_lm.pytorch.loader
variant_id: Distill_Llama_8B_Quantized_W4A16
arch: n150
status: DONE_FAIL
test_function: test_deepseek_r1_distill_llama_8b_quantized_w4a16
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: "compiler incompatibility: W4A16 compressed_tensors quantization fails in torch_xla partition_fx_graph_for_cpu_fallback (assert last_output_node is not None in fuse_by_partitions)"

# Benchmark added: test_deepseek_r1_distill_llama_8b_quantized_w4a16

## Test
tests/benchmark/test_llms.py::test_deepseek_r1_distill_llama_8b_quantized_w4a16

## Model
- HF name:    RedHatAI/DeepSeek-R1-Distill-Llama-8B-quantized.w4a16
- Loader:     third_party.tt_forge_models.deepseek_r1_distill_llama_8b_quantized_w4a16.causal_lm.pytorch.loader
- Variant:    ModelVariant.DISTILL_LLAMA_8B_QUANTIZED_W4A16

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (did not complete)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150 (n300 wormhole, single-chip assumption)

## Failure details

The model uses `compressed_tensors` W4A16 quantization (4-bit packed integer weights).
Two infra fixes were needed and applied:

1. **Missing `compressed-tensors` pip dependency** — `transformers` raises `ImportError`
   for compressed-tensors quantization configs without the package. Fixed by adding
   `compressed-tensors` to the `pyreq` field in `perf-bench-matrix.json`.

2. **Missing `hasattr` guard in `llm_benchmark.py`** — `benchmark_llm_torch_xla()`
   called `model_loader.get_weight_dtype_config_path()` unconditionally; this loader
   does not implement that method. Fixed analogously to the runner
   (`tests/runner/testers/torch/dynamic_torch_model_tester.py`) which already uses
   `hasattr`.

After both fixes, compilation fails at:

```
venv/lib/python3.12/site-packages/compressed_tensors/quantization/lifecycle/forward.py:273
  weight_data = weight.data
→ python_package/tt_torch/torch_overrides.py:9
  def __torch_function__(self, func, types, args, kwargs=None):
→ torch_xla/_dynamo/dynamo_bridge.py:791
  partition_fx_graph_for_cpu_fallback
→ torch/fx/passes/utils/fuser_utils.py:235
  assert last_output_node is not None   ← AssertionError
```

Root cause: The `compressed_tensors` W4A16 quantized forward accesses packed-weight
tensor attributes via `weight.data`, which triggers `tt_torch`'s `TorchFunctionMode`
override. The resulting graph contains an empty-output subgraph that breaks the
XLA dynamo bridge's `fuse_by_partitions`. This is a compiler-level incompatibility
between the W4A16 quantization scheme and the TT-XLA backend — not fixable from the
benchmarking infrastructure.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not complete)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        n150
- chip_count_in_system_desc:   2 (n300; single-chip n150 assumption used)
- single_chip_assumption:      true
- worker_grid_cores:           N/A
- dram_bandwidth_bytes_per_sec: N/A

### Peak FLOPs
- lofi:  N/A
- hifi2: N/A
- hifi3: N/A
- hifi4: N/A

### Compute
- total_flops:             N/A
- breakdown.matmul:        N/A
- breakdown.linear:        N/A
- breakdown.conv2d:        N/A
- breakdown.sparse_matmul: N/A

### Inputs
- count:        N/A
- memory_bytes: N/A

### KV cache
- count:        N/A
- memory_bytes: N/A
- memory_gb:    N/A

### Params
- count:                  N/A
- effective_count:        N/A
- memory_bytes:           N/A
- memory_gb:              N/A
- effective_memory_bytes: N/A
- effective_memory_gb:    N/A
- embedding_count:        N/A
- embedding_memory_bytes: N/A

### Roofline
- bound:                    N/A
- top_perf_samples_per_sec: N/A
- top_perf_time_ms:         N/A
- dram_time_ms:             N/A
- compute_time_ms_lofi:     N/A
- compute_time_ms_hifi2:    N/A
- compute_time_ms_hifi3:    N/A
- compute_time_ms_hifi4:    N/A

## Files changed
- tests/benchmark/test_llms.py (new test function)
- tests/benchmark/benchmarks/llm_benchmark.py (hasattr guard fix)
- .github/workflows/perf-bench-matrix.json (new entry + compressed-tensors dep)

## tt-forge-models submodule
no change
