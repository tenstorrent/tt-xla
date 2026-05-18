loader_path: third_party.tt_forge_models.bartowski_olmo_2_1124_7b_instruct_gguf.causal_lm.pytorch.loader
variant_id: OLMo_2_1124_7B_Instruct_GGUF
arch: p150
status: DONE_FAIL
test_function: test_bartowski_olmo_2_1124_7b_instruct_gguf
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "compiler error: 'ttnn.scaled_dot_product_attention' op Query and result must have the same element type (ttnn.TTIRToTTNNCommon pipeline failure)"

# Benchmark added: test_bartowski_olmo_2_1124_7b_instruct_gguf

## Test
tests/benchmark/test_llms.py::test_bartowski_olmo_2_1124_7b_instruct_gguf

## Model
- HF name:    bartowski/OLMo-2-1124-7B-Instruct-GGUF
- Loader:     third_party.tt_forge_models.bartowski_olmo_2_1124_7b_instruct_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.BARTOWSKI_OLMO_2_1124_7B_INSTRUCT_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (compiler failure)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details
The test failed during compilation (num_layers=1 bring-up run) with:

    loc("dot.391"): error: 'ttnn.scaled_dot_product_attention' op Query and result must have the same element type
    Failed to run TTIRToTTNNCommon pipeline

This is a TT-MLIR compiler bug: the OLMo-2 GGUF model (which uses Q4_K_M quantized
weights, de-quantized at load time) has attention layers where the query tensor
element type does not match the result element type after BFP weight conversion.
This is not fixable at the test or benchmark harness layer.

Additionally, the benchmark infrastructure was missing a guard for the optional
`get_weight_dtype_config_path` method — a general fix was applied to
`tests/benchmark/benchmarks/llm_benchmark.py` to use `getattr` with a fallback
of `None` so model loaders that don't implement this method are handled correctly.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (no perf metrics generated due to compiler failure)
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
- tests/benchmark/test_llms.py (added test_bartowski_olmo_2_1124_7b_instruct_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: getattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change (submodule at 7f719eec22)
