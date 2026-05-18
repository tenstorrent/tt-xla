loader_path: third_party.tt_forge_models.chinese_alpaca_2.causal_lm.pytorch.loader
variant_id: 7B
arch: p150
status: DONE_PASS
test_function: test_chinese_alpaca_2_7b
samples_per_second: 26.19
ttft_ms: 361.47
prefill_pcc: 0.998164
first_decode_pcc: 0.998531
top_perf_samples_per_sec: 42.5628
pct_of_target: 61.5
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: chinese_alpaca_2_7b

## Test
tests/benchmark/test_llms.py::test_chinese_alpaca_2_7b

## Model
- HF name:    hfl/chinese-alpaca-2-7b
- Loader:     third_party.tt_forge_models.chinese_alpaca_2.causal_lm.pytorch.loader
- Variant:    ModelVariant.CHINESE_ALPACA_2_7B

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  26.19
- TTFT (ms):          361.47
- Prefill PCC:        0.998164
- First decode PCC:   0.998531
- Wall clock:         0:08:24
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_chinese_alpaca_2_7b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 61.5% (26.19 / 42.5628)

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
- total_flops:             428959858816
- breakdown.matmul:        428959858816
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        1073741824
- memory_bytes: 2147483648
- memory_gb:    2

### Params
- count:                  6929256643
- effective_count:        6702764227
- memory_bytes:           7574921992
- memory_gb:              7.054695852100849
- effective_memory_bytes: 7121937160
- effective_memory_gb:    6.632820852100849
- embedding_count:        226492416
- embedding_memory_bytes: 452984832

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 42.5628
- top_perf_time_ms:         23.4947
- dram_time_ms:             15.6631
- compute_time_ms_lofi:     0.4875
- compute_time_ms_hifi2:    0.9749
- compute_time_ms_hifi3:    1.4624
- compute_time_ms_hifi4:    1.9498

## Files changed
- tests/benchmark/test_llms.py (test already existed; no changes needed)
- .github/workflows/perf-bench-matrix.json (added chinese_alpaca_2_7b entry)

## tt-forge-models submodule
no change
