loader_path: third_party.tt_forge_models.granite_3_1.causal_lm.pytorch.loader
variant_id: 3.1_8B_Instruct_Quantized_W4A16
arch: p150
status: DONE_FAIL
test_function: test_granite_3_1_8b_instruct_quantized_w4a16
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
failure_reason: "compressed-tensors W4A16 quantized forward triggers __torch_function__ override in tt_torch that breaks torch.compile FX graph partitioning: AssertionError in fuser_utils.py:insert_subgm (last_output_node is not None)"

# Benchmark added: test_granite_3_1_8b_instruct_quantized_w4a16

## Test
tests/benchmark/test_llms.py::test_granite_3_1_8b_instruct_quantized_w4a16

## Model
- HF name:    RedHatAI/granite-3.1-8b-instruct-quantized.w4a16
- Loader:     third_party.tt_forge_models.granite_3_1.causal_lm.pytorch.loader
- Variant:    ModelVariant.GRANITE_3_1_8B_INSTRUCT_QUANTIZED_W4A16

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (DONE_FAIL)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details

The model uses compressed-tensors W4A16 quantization (RedHatAI/granite-3.1-8b-instruct-quantized.w4a16),
which requires the `compressed-tensors` Python package. When loaded and compiled with `torch.compile`
and the TT backend, the quantized linear layer's `forward()` method calls `weight.data` which hits
`tt_torch/torch_overrides.py:__torch_function__`. This triggers an attempt to compile the sub-graph
via `extract_compiled_graph`, which fails in `torch/fx/passes/utils/fuser_utils.py:insert_subgm`
with:

    AssertionError: assert last_output_node is not None

This is a fundamental incompatibility between the compressed-tensors quantization runtime (which
intercepts matmul operations via `__torch_function__`) and the TT-XLA backend's FX graph
partitioning. The fix requires either:
1. Support for compressed-tensors quantized models in the TT-XLA compilation pipeline, or
2. The loader dequantizing the model weights to fp16/bf16 before passing to the benchmark harness.

Two infrastructure fixes were made:
- `tests/benchmark/benchmarks/llm_benchmark.py`: Added `hasattr` guard around
  `model_loader.get_weight_dtype_config_path()` call (general fix, not Granite-specific).
- `compressed-tensors==0.15.0.1` installed into venv (required by quantized model loader).

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        N/A
- chip_count_in_system_desc:   N/A
- single_chip_assumption:      N/A
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
- tests/benchmark/test_llms.py (added test_granite_3_1_8b_instruct_quantized_w4a16)
- tests/benchmark/benchmarks/llm_benchmark.py (hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
