loader_path: third_party.tt_forge_models.solar-10_7B.causal_lm.pytorch.loader
variant_id: Myrrh_solar_10_7b_3.0
arch: p150
status: DONE_PASS
test_function: test_myrrh_solar_10_7b_3_0
samples_per_second: 23.62
ttft_ms: 456.65
prefill_pcc: 0.998870
first_decode_pcc: 0.997877
top_perf_samples_per_sec: 30.0815
pct_of_target: 78.5
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_myrrh_solar_10_7b_3_0

## Test
tests/benchmark/test_llms.py::test_myrrh_solar_10_7b_3_0

## Model
- HF name:    MoaData/Myrrh_solar_10.7b_3.0
- Loader:     third_party.tt_forge_models.solar-10_7B.causal_lm.pytorch.loader
- Variant:    Myrrh_solar_10_7b_3.0

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  23.62
- TTFT (ms):          456.65
- Prefill PCC:        0.998870
- First decode PCC:   0.997877
- Wall clock:         ~0:12:00
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_myrrh_solar_10_7b_3_0_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 78.5% (23.62 / 30.08)

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
- total_flops:             678403506304
- breakdown.matmul:        678403506304
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        402653184
- memory_bytes: 805306368
- memory_gb:    0.75

### Params
- count:                  10731524291
- effective_count:        10600452291
- memory_bytes:           11525497608
- memory_gb:              10.7339561060071
- effective_memory_bytes: 11263353608
- effective_memory_gb:    10.4898154810071
- embedding_count:        131072000
- embedding_memory_bytes: 262144000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 30.0815
- top_perf_time_ms:         33.2430
- dram_time_ms:             22.1620
- compute_time_ms_lofi:     0.7709
- compute_time_ms_hifi2:    1.5418
- compute_time_ms_hifi3:    2.3127
- compute_time_ms_hifi4:    3.0837

## Files changed
- tests/benchmark/test_llms.py (added test_myrrh_solar_10_7b_3_0)
- .github/workflows/perf-bench-matrix.json (added myrrh_solar_10_7b_3_0 entry)
- tests/benchmark/benchmarks/llm_benchmark.py (infra fix: hasattr check for get_weight_dtype_config_path)

## tt-forge-models submodule
93218a34fc9fc6a671e0e41101da470c80891b2a → 5c3e57326233e63c36a197fe1cb520014922546c (submodule updated to latest HEAD; solar-10_7B loader with MYRRH_SOLAR_10_7B_3_0 variant was already present in both commits)
