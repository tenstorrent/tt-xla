loader_path: third_party.tt_forge_models.snowpiercer_15b_v4_gguf.causal_lm.pytorch.loader
variant_id: Q4_K_M
arch: p150
status: DONE_PASS
test_function: test_snowpiercer_15b_v4_q4_k_m_gguf
samples_per_second: 17.15457245769388
ttft_ms: 574.210029
prefill_pcc: 0.994681
first_decode_pcc: 0.995570
top_perf_samples_per_sec: 22.4819
pct_of_target: 76.3
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_snowpiercer_15b_v4_q4_k_m_gguf

## Test
tests/benchmark/test_llms.py::test_snowpiercer_15b_v4_q4_k_m_gguf

## Model
- HF name:    bartowski/TheDrummer_Snowpiercer-15B-v4-GGUF
- Loader:     third_party.tt_forge_models.snowpiercer_15b_v4_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.SNOWPIERCER_15B_V4_Q4_K_M (Q4_K_M)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  17.15
- TTFT (ms):          574.21
- Prefill PCC:        0.994681
- First decode PCC:   0.995570
- Wall clock:         0:17:41
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_snowpiercer_15b_v4_q4_k_m_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 76.3% (17.15 / 22.48)

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
- total_flops:             915364905088
- breakdown.matmul:        915364905088
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        419430400
- memory_bytes: 838860800
- memory_gb:    0.78125

### Params
- count:                  14974182595
- effective_count:        14303093955
- memory_bytes:           16539699976
- memory_gb:              15.403795965015888
- effective_memory_bytes: 15197522696
- effective_memory_gb:    14.153795965015888
- embedding_count:        671088640
- embedding_memory_bytes: 1342177280

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 22.4819
- top_perf_time_ms:         44.4803
- dram_time_ms:             29.6535
- compute_time_ms_lofi:     1.0402
- compute_time_ms_hifi2:    2.0804
- compute_time_ms_hifi3:    3.1206
- compute_time_ms_hifi4:    4.1607

## Files changed
- tests/benchmark/test_llms.py (new test function test_snowpiercer_15b_v4_q4_k_m_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: make get_weight_dtype_config_path optional)

## tt-forge-models submodule
no change
