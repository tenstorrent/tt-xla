loader_path: third_party.tt_forge_models.got_r1_14b_i1_gguf.causal_lm.pytorch.loader
variant_id: GoT_R1_14B_i1_GGUF
arch: p150
status: DONE_PASS
test_function: test_got_r1_14b_i1_gguf
samples_per_second: 14.740877863086636
ttft_ms: 539.253151
prefill_pcc: 0.997888
first_decode_pcc: 0.997380
top_perf_samples_per_sec: 23.1042
pct_of_target: 63.8
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_got_r1_14b_i1_gguf

## Test
tests/benchmark/test_llms.py::test_got_r1_14b_i1_gguf

## Model
- HF name:    mradermacher/GoT-R1-14B-i1-GGUF
- Loader:     third_party.tt_forge_models.got_r1_14b_i1_gguf.causal_lm.pytorch.loader
- Variant:    GoT_R1_14B_i1_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  14.740877863086636
- TTFT (ms):          539.253151
- Prefill PCC:        0.997888
- First decode PCC:   0.997380
- Wall clock:         0:15:24
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_got_r1_14b_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 63.8%

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
- total_flops:             895358075008
- breakdown.matmul:        895358075008
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
- count:                  14768307395
- effective_count:        13990395075
- memory_bytes:           16421018376
- memory_gb:              15.293265111744404
- effective_memory_bytes: 14865193736
- effective_memory_gb:    13.844290502369404
- embedding_count:        777912320
- embedding_memory_bytes: 1555824640

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 23.1042
- top_perf_time_ms:         43.2821
- dram_time_ms:             28.8547
- compute_time_ms_lofi:     1.0175
- compute_time_ms_hifi2:    2.0349
- compute_time_ms_hifi3:    3.0524
- compute_time_ms_hifi4:    4.0698

## Files changed
- tests/benchmark/test_llms.py (added test_got_r1_14b_i1_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed missing hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
