loader_path: third_party.tt_forge_models.compass_verifier.causal_lm.pytorch.loader
variant_id: 3B
arch: n150
status: DONE_PASS
test_function: test_compass_verifier_3b
samples_per_second: 31.249
ttft_ms: 483.682
prefill_pcc: 0.991963
first_decode_pcc: 0.998378
top_perf_samples_per_sec: 58.8915
pct_of_target: 53.1
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_compass_verifier_3b

## Test
tests/benchmark/test_llms.py::test_compass_verifier_3b

## Model
- HF name:    opencompass/CompassVerifier-3B
- Loader:     third_party.tt_forge_models.compass_verifier.causal_lm.pytorch.loader
- Variant:    ModelVariant.COMPASS_VERIFIER_3B

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  31.249
- TTFT (ms):          483.682
- Prefill PCC:        0.991963
- First decode PCC:   0.998378
- Wall clock:         0:13:39
- Hardware:           n300 (wormhole_b0, single-chip n150 assumption)

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_compass_verifier_3b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 53.1% (31.249 / 58.8915)

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
- total_flops:             197487558784
- breakdown.matmul:        185405014144
- breakdown.linear:        12082544640
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        75497472
- memory_bytes: 150994944
- memory_gb:    0.140625

### Params
- count:                  3397103811
- effective_count:        3085938883
- memory_bytes:           3901367048
- memory_gb:              3.633431203663349
- effective_memory_bytes: 3279037192
- effective_memory_gb:    3.053841359913349
- embedding_count:        311164928
- embedding_memory_bytes: 622329856

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 58.8915
- top_perf_time_ms:         16.9804
- dram_time_ms:             11.3202
- compute_time_ms_lofi:     0.7714
- compute_time_ms_hifi2:    1.5429
- compute_time_ms_hifi3:    2.3143
- compute_time_ms_hifi4:    3.0857

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
