loader_path: third_party.tt_forge_models.chinese_alpaca_2.causal_lm.pytorch.loader
variant_id: 7B
arch: n150
status: DONE_PASS
test_function: test_chinese_alpaca_2_7b
samples_per_second: 14.975
ttft_ms: 780.034
prefill_pcc: 0.998198
first_decode_pcc: 0.998278
top_perf_samples_per_sec: 23.9416
pct_of_target: 62.5
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_chinese_alpaca_2_7b

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
- Sample per second:  14.975
- TTFT (ms):          780.034
- Prefill PCC:        0.998198
- First decode PCC:   0.998278
- Wall clock:         0:16:31
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_chinese_alpaca_2_7b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 62.5%

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
- top_perf_samples_per_sec: 23.9416
- top_perf_time_ms:         41.7683
- dram_time_ms:             27.8456
- compute_time_ms_lofi:     1.6756
- compute_time_ms_hifi2:    3.3512
- compute_time_ms_hifi3:    5.0269
- compute_time_ms_hifi4:    6.7025

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py

## tt-forge-models submodule
no change
