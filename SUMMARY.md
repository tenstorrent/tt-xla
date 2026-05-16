loader_path: third_party.tt_forge_models.aidc_llm_laos_4b_gguf.causal_lm.pytorch.loader
variant_id: Q4_K_M
arch: n150
status: DONE_FAIL
test_function: test_aidc_llm_laos_4b
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
failure_reason: "device hardware error: ARC startup timeout on first init (postcode=0xc0de0042); tt-smi --reset failed with Ethernet core service timeout — requires machine reboot"

# Benchmark added: test_aidc_llm_laos_4b

## Test
tests/benchmark/test_llms.py::test_aidc_llm_laos_4b

## Model
- HF name:    mradermacher/aidc-llm-laos-4b-GGUF
- Loader:     third_party.tt_forge_models.aidc_llm_laos_4b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.AIDC_LLM_LAOS_4B_Q4_K_M (value: "Q4_K_M")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A — test not reached (device error)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150 (wormhole_b0)

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A — perf metrics not captured (device did not initialize)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        n150 (wormhole_b0)
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
- tests/benchmark/test_llms.py (added test_aidc_llm_laos_4b)
- .github/workflows/perf-bench-matrix.json (added aidc_llm_laos_4b_gguf entry)

## tt-forge-models submodule
no change
