loader_path: third_party.tt_forge_models.rugpt3small.causal_lm.pytorch.loader
variant_id: ruGPT3_Small
arch: p150
status: DONE_PASS
test_function: test_rugpt3_small
samples_per_second: 162.73
ttft_ms: 85.883
prefill_pcc: 0.999018
first_decode_pcc: 0.999275
top_perf_samples_per_sec: 1661.9600
pct_of_target: 9.8
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: rugpt3_small

## Test
tests/benchmark/test_llms.py::test_rugpt3_small

## Model
- HF name:    ai-forever/rugpt3small_based_on_gpt2
- Loader:     third_party.tt_forge_models.rugpt3small.causal_lm.pytorch.loader
- Variant:    ModelVariant.RUGPT3_SMALL

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  162.73
- TTFT (ms):          85.883
- Prefill PCC:        0.999018
- First decode PCC:   0.999275
- Wall clock:         0:02:12
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_rugpt3_small_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 9.8% (162.73 / 1661.96)

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
- total_flops:             158180966400
- breakdown.matmul:        49411522560
- breakdown.linear:        108769443840
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        660
- memory_bytes: 2640

### KV cache
- count:        75497472
- memory_bytes: 150994944
- memory_gb:    0.140625

### Params
- count:                  163834500
- effective_count:        123658884
- memory_bytes:           212018828
- memory_gb:              0.1974579207599163
- effective_memory_bytes: 131667596
- effective_memory_gb:    0.12262500450015068
- embedding_count:        40175616
- embedding_memory_bytes: 80351232

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 1661.9600
- top_perf_time_ms:         0.6017
- dram_time_ms:             0.4011
- compute_time_ms_lofi:     0.1798
- compute_time_ms_hifi2:    0.3595
- compute_time_ms_hifi3:    0.5393
- compute_time_ms_hifi4:    0.7190

## Files changed
- tests/benchmark/test_llms.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
