loader_path: third_party.tt_forge_models.llama_300m_v5.causal_lm.pytorch.loader
variant_id: llama_300m_v5_window_4
arch: p150
status: DONE_PASS
test_function: test_llama_300m_v5_window_4
samples_per_second: 230.2023828261318
ttft_ms: 69.015907
prefill_pcc: 0.999210
first_decode_pcc: 0.998013
top_perf_samples_per_sec: 795.5241
pct_of_target: 28.9
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: llama_300m_v5_window_4

## Test
tests/benchmark/test_llms.py::test_llama_300m_v5_window_4

## Model
- HF name:    deqing/llama-300M-v5-window_4
- Loader:     third_party.tt_forge_models.llama_300m_v5.causal_lm.pytorch.loader
- Variant:    ModelVariant.LLAMA_300M_V5_WINDOW_4

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  230.2023828261318
- TTFT (ms):          69.015907
- Prefill PCC:        0.999210
- First decode PCC:   0.998013
- Wall clock:         0:01:55
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_llama_300m_v5_window_4_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 28.9%

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
- total_flops:             368729654400
- breakdown.matmul:        368729654400
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        594
- memory_bytes: 2376

### KV cache
- count:        50331648
- memory_bytes: 100663296
- memory_gb:    0.09375

### Params
- count:                  451437731
- effective_count:        320103587
- memory_bytes:           602802824
- memory_gb:              0.5614038780331612
- effective_memory_bytes: 340134536
- effective_memory_gb:    0.31677497178316116
- embedding_count:        131334144
- embedding_memory_bytes: 262668288

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 795.5241
- top_perf_time_ms:         1.2570
- dram_time_ms:             0.7461
- compute_time_ms_lofi:     0.4190
- compute_time_ms_hifi2:    0.8380
- compute_time_ms_hifi3:    1.2570
- compute_time_ms_hifi4:    1.6760

## Files changed
- tests/benchmark/test_llms.py (new test function test_llama_300m_v5_window_4)
- tests/benchmark/benchmarks/llm_benchmark.py (guard get_weight_dtype_config_path with hasattr check)
- .github/workflows/perf-bench-matrix.json (new entry llama_300m_v5_window_4)

## tt-forge-models submodule
no change
