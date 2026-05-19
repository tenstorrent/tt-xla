loader_path: third_party.tt_forge_models.mradermacher_ml_distilled_i1_gguf.causal_lm.pytorch.loader
variant_id: ML_DISTILLED_I1_GGUF
arch: p150
status: DONE_PASS
test_function: test_mradermacher_ml_distilled_i1_gguf
samples_per_second: 19.52
ttft_ms: 421.37
prefill_pcc: 0.998222
first_decode_pcc: 0.998280
top_perf_samples_per_sec: 25.1586
pct_of_target: 77.6
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: mradermacher_ml_distilled_i1_gguf

## Test
tests/benchmark/test_llms.py::test_mradermacher_ml_distilled_i1_gguf

## Model
- HF name:    mradermacher/ML-Distilled-i1-GGUF
- Loader:     third_party.tt_forge_models.mradermacher_ml_distilled_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.ML_DISTILLED_I1_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  19.52
- TTFT (ms):          421.37
- Prefill PCC:        0.998222
- First decode PCC:   0.998280
- Wall clock:         0:13:41
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_mradermacher_ml_distilled_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 77.6% (19.52 / 25.16)

### System
- arch:                        blackhole
- chip_count_in_system_desc:   1
- single_chip_assumption:      True
- worker_grid_cores:           110
- dram_bandwidth_bytes_per_sec: 512000000000

### Peak FLOPs
- lofi:  880000000000000
- hifi2: 440000000000000
- hifi3: 293333333333333
- hifi4: 220000000000000

### Compute
- total_flops:             820489748608
- breakdown.matmul:        820489748608
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        335544320
- memory_bytes: 671088640
- memory_gb:    0.625

### Params
- count:                  13477237955
- effective_count:        12820567235
- memory_bytes:           14935583496
- memory_gb:              13.90984607487917
- effective_memory_bytes: 13622242056
- effective_memory_gb:    12.68670154362917
- embedding_count:        656670720
- embedding_memory_bytes: 1313341440

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 25.1586
- top_perf_time_ms:         39.7478
- dram_time_ms:             26.4985
- compute_time_ms_lofi:     0.9324
- compute_time_ms_hifi2:    1.8647
- compute_time_ms_hifi3:    2.7971
- compute_time_ms_hifi4:    3.7295

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (general infrastructure fix: added hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
