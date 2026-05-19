loader_path: third_party.tt_forge_models.smollm3.causal_lm.pytorch.loader
variant_id: SmolLM3_3B
arch: p150
status: DONE_PASS
test_function: test_smollm3_3b
samples_per_second: 57.620162574681075
ttft_ms: 243.757881
prefill_pcc: 0.997472
first_decode_pcc: 0.996777
top_perf_samples_per_sec: 791.1325
pct_of_target: 7.3
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: smollm3_3b

## Test
tests/benchmark/test_llms.py::test_smollm3_3b

## Model
- HF name:    HuggingFaceTB/SmolLM3-3B
- Loader:     third_party.tt_forge_models.smollm3.causal_lm.pytorch.loader
- Variant:    SmolLM3_3B

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  57.620162574681075
- TTFT (ms):          243.757881
- Prefill PCC:        0.997472
- First decode PCC:   0.996777
- Wall clock:         0:07:36
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_smollm3_3b_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 7.3%

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
- total_flops:             370776475776
- breakdown.matmul:        370776475776
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        561
- memory_bytes: 2244

### KV cache
- count:        4194304
- memory_bytes: 8388608
- memory_gb:    0.0078125

### Params
- count:                  603461827
- effective_count:        340793539
- memory_bytes:           887436040
- memory_gb:              0.8264892175793648
- effective_memory_bytes: 362099464
- effective_memory_gb:    0.3372314050793648
- embedding_count:        262668288
- embedding_memory_bytes: 525336576

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 791.1325
- top_perf_time_ms:         1.2640
- dram_time_ms:             0.6949
- compute_time_ms_lofi:     0.4213
- compute_time_ms_hifi2:    0.8427
- compute_time_ms_hifi3:    1.2640
- compute_time_ms_hifi4:    1.6853

## Files changed
- tests/benchmark/test_llms.py (added test_smollm3_3b)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json (added smollm3_3b entry)

## tt-forge-models submodule
no change
