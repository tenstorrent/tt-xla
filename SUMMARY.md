loader_path: third_party.tt_forge_models.barbot_8b_v1_i1_gguf.causal_lm.pytorch.loader
variant_id: BARBOT_8B_V1_I1_GGUF
arch: p150
status: DONE_PASS
test_function: test_barbot_8b_v1_i1_gguf
samples_per_second: 33.68456157702509
ttft_ms: 329.053205
prefill_pcc: 0.999116
first_decode_pcc: 0.998466
top_perf_samples_per_sec: 42.5628
pct_of_target: 79.1
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_barbot_8b_v1_i1_gguf

## Test
tests/benchmark/test_llms.py::test_barbot_8b_v1_i1_gguf

## Model
- HF name:    mradermacher/Barbot-8B-v1-i1-GGUF
- Loader:     third_party.tt_forge_models.barbot_8b_v1_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.BARBOT_8B_V1_I1_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  33.68456157702509
- TTFT (ms):          329.053205
- Prefill PCC:        0.999116
- First decode PCC:   0.998466
- Wall clock:         0:09:40
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_barbot_8b_v1_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 79.1% (33.68 / 42.56)

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
- total_flops:             480499466368
- breakdown.matmul:        480499466368
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        268435456
- memory_bytes: 536870912
- memory_gb:    0.5

### Params
- count:                  8036552899
- effective_count:        7508070595
- memory_bytes:           9034539784
- memory_gb:              8.41407085210085
- effective_memory_bytes: 7977575176
- effective_memory_gb:    7.429695852100849
- embedding_count:        528482304
- embedding_memory_bytes: 1056964608

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 42.5628
- top_perf_time_ms:         23.4947
- dram_time_ms:             15.6631
- compute_time_ms_lofi:     0.5460
- compute_time_ms_hifi2:    1.0920
- compute_time_ms_hifi3:    1.6381
- compute_time_ms_hifi4:    2.1841

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: guarded get_weight_dtype_config_path call with hasattr)

## tt-forge-models submodule
no change
