loader_path: third_party.tt_forge_models.mellum_4b_base.causal_lm.pytorch.loader
variant_id: mellum_4b_base
arch: p150
status: DONE_PASS
test_function: test_mellum_4b_base
samples_per_second: 36.44
ttft_ms: 282.69
prefill_pcc: 0.999758
first_decode_pcc: 0.999397
top_perf_samples_per_sec: 74.01
pct_of_target: 49.2
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_mellum_4b_base

## Test
tests/benchmark/test_llms.py::test_mellum_4b_base

## Model
- HF name:    JetBrains/Mellum-4b-base
- Loader:     third_party.tt_forge_models.mellum_4b_base.causal_lm.pytorch.loader
- Variant:    mellum_4b_base

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  36.44
- TTFT (ms):          282.69
- Prefill PCC:        0.999758
- First decode PCC:   0.999397
- Wall clock:         0:06:01
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_mellum_4b_base_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 49.2% (36.44 / 74.01)

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
- total_flops:             237892534400
- breakdown.matmul:        237892534400
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        754974720
- memory_bytes: 1509949440
- memory_gb:    1.40625

### Params
- count:                  4019248323
- effective_count:        3717258435
- memory_bytes:           4553743112
- memory_gb:              4.241003759205341
- effective_memory_bytes: 3949763336
- effective_memory_gb:    3.6785037592053413
- embedding_count:        301989888
- embedding_memory_bytes: 603979776

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 74.0100
- top_perf_time_ms:         13.5117
- dram_time_ms:             9.0078
- compute_time_ms_lofi:     0.2703
- compute_time_ms_hifi2:    0.5407
- compute_time_ms_hifi3:    0.8110
- compute_time_ms_hifi4:    1.0813

## Files changed
- tests/benchmark/test_llms.py (added test_mellum_4b_base)
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: guard get_weight_dtype_config_path with hasattr check)

## tt-forge-models submodule
no change
