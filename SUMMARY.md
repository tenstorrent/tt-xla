loader_path: third_party.tt_forge_models.anakin87_llama_3_8b_ita_slerp.causal_lm.pytorch.loader
variant_id: Llama_3_8B_ita_slerp
arch: n150
status: DONE_FAIL
test_function: test_anakin87_llama_3_8b_ita_slerp
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
failure_reason: "device hardware failure: Timeout waiting for Ethernet core service remote IO request flush — device inaccessible after multiple tt-smi resets"

# Benchmark added: test_anakin87_llama_3_8b_ita_slerp

## Test
tests/benchmark/test_llms.py::test_anakin87_llama_3_8b_ita_slerp

## Model
- HF name:    anakin87/Llama-3-8b-ita-slerp
- Loader:     third_party.tt_forge_models.anakin87_llama_3_8b_ita_slerp.causal_lm.pytorch.loader
- Variant:    ModelVariant.LLAMA_3_8B_ITA_SLERP

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (device hardware failure)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150 (Wormhole n300)

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A — test could not complete due to device failure
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

## tt-forge-models submodule
no change
