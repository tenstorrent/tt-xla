loader_path: third_party.tt_forge_models.tiny_random_olmo.causal_lm.pytorch.loader
variant_id: tiny_random_olmo
arch: p150
status: DONE_FAIL
test_function: test_tiny_random_olmo
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
failure_reason: "compiler error: 'ttnn.scaled_dot_product_attention' op Query and result must have the same element type in TTIRToTTNNCommon pipeline"

# Benchmark added: test_tiny_random_olmo

## Test
tests/benchmark/test_llms.py::test_tiny_random_olmo

## Model
- HF name:    katuni4ka/tiny-random-olmo-hf
- Loader:     third_party.tt_forge_models.tiny_random_olmo.causal_lm.pytorch.loader
- Variant:    ModelVariant.TINY_RANDOM_OLMO

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test failed)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details
The test fails at warmup (prefill graph compilation) with:

    loc("dot.359"): error: 'ttnn.scaled_dot_product_attention' op Query and result must have the same element type
    Failed to run TTIRToTTNNCommon pipeline

This is a compiler bug in the TTIR->TTNN lowering pipeline. The OLMo attention
computation produces a type mismatch between the query tensor and the output
of ttnn.scaled_dot_product_attention. This is not fixable from the test side.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not complete)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        p150 (Blackhole)
- chip_count_in_system_desc:   1
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
- tests/benchmark/test_llms.py (added test_tiny_random_olmo)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr check)

## tt-forge-models submodule
no change
