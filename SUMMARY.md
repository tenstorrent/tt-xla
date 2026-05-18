loader_path: third_party.tt_forge_models.gpt_neo.causal_lm.pytorch.loader
variant_id: TinyStories_33M
arch: p150
status: DONE_PASS
test_function: test_gpt_neo_tinystories_33m
samples_per_second: 250.95
ttft_ms: 34.087
prefill_pcc: 0.973031
first_decode_pcc: 0.943052
top_perf_samples_per_sec: 3593.51
pct_of_target: 7.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: null
failure_reason: null

# Benchmark added: test_gpt_neo_tinystories_33m

## Test
tests/benchmark/test_llms.py::test_gpt_neo_tinystories_33m

## Model
- HF name:    roneneldan/TinyStories-33M
- Loader:     third_party.tt_forge_models.gpt_neo.causal_lm.pytorch.loader
- Variant:    ModelVariant.TINYSTORIES_33M

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: none (disabled — bfp_bf8 caused first-decode PCC failure: 0.938373 < 0.94)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  250.95
- TTFT (ms):          34.087
- Prefill PCC:        0.973031
- First decode PCC:   0.943052
- Wall clock:         0:00:49
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_gpt_neo_tinystories_33m_perf_metrics_2.json
Achieved vs top_perf_samples_per_sec: 7.0% (250.95 / 3593.51)

### System
- arch:                        blackhole
- chip_count_in_system_desc:   1
- single_chip_assumption:      true
- worker_grid_cores:           110
- dram_bandwidth_bytes_per_sec: 512000000000

### Peak FLOPs
- lofi:  880000000000000
- hifi2: 440000000000000
- hifi3: 293333333333333
- hifi4: 220000000000000

### Compute
- total_flops:             4333092864
- breakdown.matmul:        2973548544
- breakdown.linear:        1359544320
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        25165824
- memory_bytes: 50331648
- memory_gb:    0.046875

### Params
- count:                  107112071
- effective_count:        66941831
- memory_bytes:           214225432
- memory_gb:              0.19951298087835312
- effective_memory_bytes: 133884952
- effective_memory_gb:    0.12469007819890976
- embedding_count:        40170240
- embedding_memory_bytes: 80340480

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 3593.51
- top_perf_time_ms:         0.2783
- dram_time_ms:             0.1855
- compute_time_ms_lofi:     0.004924
- compute_time_ms_hifi2:    0.009848
- compute_time_ms_hifi3:    0.014772
- compute_time_ms_hifi4:    0.019696

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
