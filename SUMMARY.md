loader_path: third_party.tt_forge_models.l_m_chat.causal_lm.pytorch.loader
variant_id: Small
arch: p150
status: DONE_PASS
test_function: test_l_m_chat_small
samples_per_second: 25.476882529014308
ttft_ms: 448.782606
prefill_pcc: 0.992427
first_decode_pcc: 0.993280
top_perf_samples_per_sec: 99.7047
pct_of_target: 25.6
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: l_m_chat_small

## Test
tests/benchmark/test_llms.py::test_l_m_chat_small

## Model
- HF name:    Artples/L-MChat-Small
- Loader:     third_party.tt_forge_models.l_m_chat.causal_lm.pytorch.loader
- Variant:    ModelVariant.L_MCHAT_SMALL

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  25.476882529014308
- TTFT (ms):          448.782606
- Prefill PCC:        0.992427
- First decode PCC:   0.993280
- Wall clock:         0:05:38
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_l_m_chat_small_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 25.6% (25.48 / 99.70)

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
- total_flops:             169475112992
- breakdown.matmul:        32
- breakdown.linear:        169475112960
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        671088640
- memory_bytes: 1342177280
- memory_gb:    1.25

### Params
- count:                  2779683987
- effective_count:        2648611987
- memory_bytes:           3077192264
- memory_gb:              2.8658586218953133
- effective_memory_bytes: 2815048264
- effective_memory_gb:    2.6217179968953133
- embedding_count:        131072000
- embedding_memory_bytes: 262144000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 99.7047
- top_perf_time_ms:         10.0296
- dram_time_ms:             6.6864
- compute_time_ms_lofi:     0.1926
- compute_time_ms_hifi2:    0.3852
- compute_time_ms_hifi3:    0.5778
- compute_time_ms_hifi4:    0.7703

## Files changed
- tests/benchmark/test_llms.py (added test_l_m_chat_small)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: use getattr for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added l_m_chat_small entry)

## tt-forge-models submodule
no change
