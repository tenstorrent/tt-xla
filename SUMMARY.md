loader_path: third_party.tt_forge_models.llmscience.causal_lm.pytorch.loader
variant_id: llmscience
arch: p150
status: DONE_FAIL
test_function: test_llmscience
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
failure_reason: "transformers API incompatibility: StaticCache has no attribute 'has_previous_state' in qwen3_5/modeling_qwen3_5.py:525"

# Benchmark added: test_llmscience

## Test
tests/benchmark/test_llms.py::test_llmscience

## Model
- HF name:    LauraRuis/llmscience
- Loader:     third_party.tt_forge_models.llmscience.causal_lm.pytorch.loader
- Variant:    ModelVariant.LLMSCIENCE

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
The test failed with:
  AttributeError: 'StaticCache' object has no attribute 'has_previous_state'

The LauraRuis/llmscience model uses the Qwen3.5 architecture
(transformers/models/qwen3_5/modeling_qwen3_5.py). The Qwen3.5 forward
pass at line 525 checks `cache_params.has_previous_state`, but the
`StaticCache` class in the installed transformers version does not expose
this attribute. This is a transformers API incompatibility — the model
loader assumes a transformers version that added `has_previous_state` to
`StaticCache`, but the current environment has an older version without it.

This is out of scope for the benchmark infrastructure to fix; the fix
belongs in the model loader / tt-forge-models repo or requires a
transformers version update.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not complete)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        p150
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

## tt-forge-models submodule
no change
