loader_path: third_party.tt_forge_models.internlm.causal_lm.pytorch.loader
variant_id: tiny_random
arch: p150
status: DONE_FAIL
test_function: test_internlm_tiny_random
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
failure_reason: "TypeError: InternLMForCausalLM.forward() got an unexpected keyword argument 'cache_position' (loader forward signature incompatible with transformers 5.2)"

# Benchmark added: test_internlm_tiny_random

## Test
tests/benchmark/test_llms.py::test_internlm_tiny_random

## Model
- HF name:    optimum-intel-internal-testing/tiny-random-internlm
- Loader:     third_party.tt_forge_models.internlm.causal_lm.pytorch.loader
- Variant:    ModelVariant.TINY_RANDOM

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure
The test failed at the CPU prefill step with:
  TypeError: InternLMForCausalLM.forward() got an unexpected keyword argument 'cache_position'

This is a loader incompatibility: the InternLM custom modeling code
(downloaded from optimum-intel-internal-testing/tiny-random-internlm via
trust_remote_code=True) does not accept the `cache_position` keyword
argument that transformers 5.2 passes to the forward() method. This
requires a fix in the loader / model code in tt-forge-models, not in
this benchmark harness.

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
- tests/benchmark/test_llms.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
