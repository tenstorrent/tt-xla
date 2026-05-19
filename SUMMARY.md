loader_path: third_party.tt_forge_models.karasu.causal_lm.pytorch.loader
variant_id: 1_1B
arch: p150
status: DONE_PASS
test_function: test_karasu_1_1b
samples_per_second: 91.03
ttft_ms: 104.16
prefill_pcc: 0.997385
first_decode_pcc: 0.998351
top_perf_samples_per_sec: 306.2878
pct_of_target: 29.7
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: karasu_1_1b

## Test
tests/benchmark/test_llms.py::test_karasu_1_1b

## Model
- HF name:    lightblue/karasu-1.1B
- Loader:     third_party.tt_forge_models.karasu.causal_lm.pytorch.loader
- Variant:    ModelVariant.KARASU_1_1B (1_1B)

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=2 fails with ttnn.paged_update_cache requiring sharded input tensor;
optimization_level=1 is stable and passes PCC.

## Measured (full model, defaults)
- Sample per second:  91.03
- TTFT (ms):          104.16
- Prefill PCC:        0.997385
- First decode PCC:   0.998351
- Wall clock:         0:01:10
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_karasu_1_1b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 29.7% (91.03 / 306.29)

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
- total_flops:             66202894400
- breakdown.matmul:        66202894400
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        46137344
- memory_bytes: 92274688
- memory_gb:    0.0859375

### Params
- count:                  1100048547
- effective_count:        1034512547
- memory_bytes:           1230328456
- memory_gb:              1.1458326652646065
- effective_memory_bytes: 1099256456
- effective_memory_gb:    1.0237623527646065
- embedding_count:        65536000
- embedding_memory_bytes: 131072000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 306.2878
- top_perf_time_ms:         3.2649
- dram_time_ms:             2.1766
- compute_time_ms_lofi:     0.0752
- compute_time_ms_hifi2:    0.1505
- compute_time_ms_hifi3:    0.2257
- compute_time_ms_hifi4:    0.3009

## Files changed
- tests/benchmark/test_llms.py (added test_karasu_1_1b)
- tests/benchmark/benchmarks/llm_benchmark.py (guard get_weight_dtype_config_path with hasattr check)
- .github/workflows/perf-bench-matrix.json (added karasu_1_1b entry)

## tt-forge-models submodule
no change
