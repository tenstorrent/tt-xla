loader_path: third_party.tt_forge_models.bella_bartender_heretic_1b_i1_gguf.causal_lm.pytorch.loader
variant_id: 1B_i1_GGUF
arch: n150
status: DONE_PASS
test_function: test_bella_bartender_heretic_1b_i1_gguf
samples_per_second: 66.44
ttft_ms: 246.67
prefill_pcc: 0.998131
first_decode_pcc: 0.997600
top_perf_samples_per_sec: 142.8954
pct_of_target: 46.5
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_bella_bartender_heretic_1b_i1_gguf

## Test
tests/benchmark/test_llms.py::test_bella_bartender_heretic_1b_i1_gguf

## Model
- HF name:    mradermacher/bella-bartender-heretic-1b-i1-GGUF
- Loader:     third_party.tt_forge_models.bella_bartender_heretic_1b_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.BELLA_BARTENDER_HERETIC_1B_I1_GGUF (1B_i1_GGUF)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  66.44
- TTFT (ms):          246.67
- Prefill PCC:        0.998131
- First decode PCC:   0.997600
- Wall clock:         0:06:16
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_bella_bartender_heretic_1b_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 46.5% (66.44 / 142.90)

### System
- arch:                        wormhole_b0
- chip_count_in_system_desc:   2
- single_chip_assumption:      True
- worker_grid_cores:           64
- dram_bandwidth_bytes_per_sec: 288000000000

### Peak FLOPs
- lofi:  256000000000000
- hifi2: 128000000000000
- hifi3: 85333333333333
- hifi4: 64000000000000

### Compute
- total_flops:             79087796288
- breakdown.matmul:        79087796288
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        67108864
- memory_bytes: 134217728
- memory_gb:    0.125

### Params
- count:                  1498482851
- effective_count:        1235814563
- memory_bytes:           1838453384
- memory_gb:              1.712
- effective_memory_bytes: 1313116808
- effective_memory_gb:    1.223
- embedding_count:        262668288
- embedding_memory_bytes: 525336576

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 142.8954
- top_perf_time_ms:         6.9981
- dram_time_ms:             4.6654
- compute_time_ms_lofi:     0.3089
- compute_time_ms_hifi2:    0.6179
- compute_time_ms_hifi3:    0.9268
- compute_time_ms_hifi4:    1.2357

## Files changed
- tests/benchmark/test_llms.py (added test_bella_bartender_heretic_1b_i1_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (graceful fallback for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added bella_bartender_heretic_1b_i1_gguf entry)

## tt-forge-models submodule
no change
