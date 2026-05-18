loader_path: third_party.tt_forge_models.bella_bartender_heretic.causal_lm.pytorch.loader
variant_id: BELLA_BARTENDER_HERETIC_1B
arch: p150
status: DONE_PASS
test_function: test_bella_bartender_heretic_1b
samples_per_second: 123.96
ttft_ms: 100.39
prefill_pcc: 0.997980
first_decode_pcc: 0.998218
top_perf_samples_per_sec: 254.0363
pct_of_target: 48.8
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_bella_bartender_heretic_1b

## Test
tests/benchmark/test_llms.py::test_bella_bartender_heretic_1b

## Model
- HF name:    juiceb0xc0de/bella-bartender-heretic-1b
- Loader:     third_party.tt_forge_models.bella_bartender_heretic.causal_lm.pytorch.loader
- Variant:    ModelVariant.BELLA_BARTENDER_HERETIC_1B

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  123.96
- TTFT (ms):          100.39
- Prefill PCC:        0.997980
- First decode PCC:   0.998218
- Wall clock:         0:03:11
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_bella_bartender_heretic_1b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 48.8% (123.96 / 254.04)

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
- total_flops:             79087796288
- breakdown.matmul:        79087796288
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        67108864
- memory_bytes: 134217728
- memory_gb:    0.125

### Params
- count:                  1498482851
- effective_count:        1235814563
- memory_bytes:           1838453384
- memory_gb:              1.712193138897419
- effective_memory_bytes: 1313116808
- effective_memory_gb:    1.222935326397419
- embedding_count:        262668288
- embedding_memory_bytes: 525336576

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 254.0363
- top_perf_time_ms:         3.9364
- dram_time_ms:             2.6243
- compute_time_ms_lofi:     0.0899
- compute_time_ms_hifi2:    0.1797
- compute_time_ms_hifi3:    0.2696
- compute_time_ms_hifi4:    0.3595

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py

## tt-forge-models submodule
no change
