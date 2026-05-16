loader_path: third_party.tt_forge_models.afmck_testing_llama_tiny.causal_lm.pytorch.loader
variant_id: testing_llama_tiny
arch: n150
status: DONE_FAIL
test_function: test_afmck_testing_llama_tiny
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
failure_reason: "hardware device unavailable: Timeout waiting for Ethernet core service remote IO request flush on N300 board — ethernet link between chips is broken, device cannot initialize even after multiple tt-smi resets"

# Benchmark added: test_afmck_testing_llama_tiny

## Test
tests/benchmark/test_llms.py::test_afmck_testing_llama_tiny

## Model
- HF name:    afmck/testing-llama-tiny
- Loader:     third_party.tt_forge_models.afmck_testing_llama_tiny.causal_lm.pytorch.loader
- Variant:    ModelVariant.TESTING_LLAMA_TINY

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (device unavailable)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n300 (wormhole_b0)

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A — test could not run due to hardware failure
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
- tests/benchmark/test_llms.py (added test_afmck_testing_llama_tiny)
- .github/workflows/perf-bench-matrix.json (added afmck_testing_llama_tiny entry)

## tt-forge-models submodule
no change — submodule HEAD remains at 7a0b51fd79
