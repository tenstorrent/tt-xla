loader_path: third_party.tt_forge_models.configurable_yi.causal_lm.pytorch.loader
variant_id: 1.5_9B_Chat
arch: n150
status: DONE_PASS
test_function: test_configurable_yi_1_5_9b_chat
samples_per_second: 15.6235
ttft_ms: 874.540243
prefill_pcc: 0.999215
first_decode_pcc: 0.999272
top_perf_samples_per_sec: 21.2328
pct_of_target: 73.6
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_configurable_yi_1_5_9b_chat

## Test
tests/benchmark/test_llms.py::test_configurable_yi_1_5_9b_chat

## Model
- HF name:    vicgalle/Configurable-Yi-1.5-9B-Chat
- Loader:     third_party.tt_forge_models.configurable_yi.causal_lm.pytorch.loader
- Variant:    ModelVariant.CONFIGURABLE_YI_1_5_9B_CHAT

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  15.6235
- TTFT (ms):          874.540243
- Prefill PCC:        0.999215
- First decode PCC:   0.999272
- Wall clock:         0:34:01
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_configurable_yi_1_5_9b_chat_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 73.6%

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
- total_flops:             548279419008
- breakdown.matmul:        548279419008
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        201326592
- memory_bytes: 402653184
- memory_gb:    0.375

### Params
- count:                  8829407427
- effective_count:        8567263427
- memory_bytes:           9627378440
- memory_gb:              8.96619487553835
- effective_memory_bytes: 9103090440
- effective_memory_gb:    8.47791362553835
- embedding_count:        262144000
- embedding_memory_bytes: 524288000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 21.2328
- top_perf_time_ms:         47.0969
- dram_time_ms:             31.3979
- compute_time_ms_lofi:     2.1417
- compute_time_ms_hifi2:    4.2834
- compute_time_ms_hifi3:    6.4251
- compute_time_ms_hifi4:    8.5669

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
