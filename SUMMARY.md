loader_path: third_party.tt_forge_models.athena_1_3b_i1_gguf.causal_lm.pytorch.loader
variant_id: 3B_I1_GGUF
arch: n150
status: DONE_PASS
test_function: test_athena_1_3b_i1_gguf
samples_per_second: 30.197
ttft_ms: 465.311
prefill_pcc: 0.996012
first_decode_pcc: 0.998440
top_perf_samples_per_sec: 58.8915
pct_of_target: 51.3
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_athena_1_3b_i1_gguf

## Test
tests/benchmark/test_llms.py::test_athena_1_3b_i1_gguf

## Model
- HF name:    mradermacher/Athena-1-3B-i1-GGUF
- Loader:     third_party.tt_forge_models.athena_1_3b_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.ATHENA_1_3B_I1_GGUF (3B_I1_GGUF)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  30.197
- TTFT (ms):          465.311
- Prefill PCC:        0.996012
- First decode PCC:   0.998440
- Wall clock:         0:14:44
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_athena_1_3b_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 51.3%

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
- total_flops:             197487558784
- breakdown.matmul:        185405014144
- breakdown.linear:        12082544640
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        75497472
- memory_bytes: 150994944
- memory_gb:    0.140625

### Params
- count:                  3397103811
- effective_count:        3085938883
- memory_bytes:           3901367048
- memory_gb:              3.633431203663349
- effective_memory_bytes: 3279037192
- effective_memory_gb:    3.053841359913349
- embedding_count:        311164928
- embedding_memory_bytes: 622329856

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 58.8915
- top_perf_time_ms:         16.9804
- dram_time_ms:             11.3202
- compute_time_ms_lofi:     0.7714
- compute_time_ms_hifi2:    1.5429
- compute_time_ms_hifi3:    2.3143
- compute_time_ms_hifi4:    3.0857

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: graceful fallback for loaders missing get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
