loader_path: third_party.tt_forge_models.blossom.causal_lm.pytorch.loader
variant_id: Blossom_v5.1_9B
arch: n150
status: DONE_PASS
test_function: test_blossom_v5_1_9b
samples_per_second: 15.428257949532318
ttft_ms: 870.293855
prefill_pcc: 0.998616
first_decode_pcc: 0.999036
top_perf_samples_per_sec: 21.2328
pct_of_target: 72.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_blossom_v5_1_9b

## Test
tests/benchmark/test_llms.py::test_blossom_v5_1_9b

## Model
- HF name:    Azure99/blossom-v5.1-9b
- Loader:     third_party.tt_forge_models.blossom.causal_lm.pytorch.loader
- Variant:    ModelVariant.BLOSSOM_V5_1_9B

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  15.428257949532318
- TTFT (ms):          870.293855
- Prefill PCC:        0.998616
- First decode PCC:   0.999036
- Wall clock:         0:22:15
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_blossom_v5_1_9b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 72.7%

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
- total_flops:             548279419008
- breakdown.matmul:        548279419008
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        201326592
- memory_bytes: 402653184
- memory_gb:    0.375

### Params
- count:                  8829407427
- effective_count:        8567263427
- memory_bytes:           9627378440
- memory_gb:              8.96619487553835
- effective_memory_bytes: 9103090440
- effective_memory_gb:    8.47791362553835
- embedding_count:        262144000
- embedding_memory_bytes: 524288000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 21.2328
- top_perf_time_ms:         47.0969
- dram_time_ms:             31.3979
- compute_time_ms_lofi:     2.1417
- compute_time_ms_hifi2:    4.2834
- compute_time_ms_hifi3:    6.4251
- compute_time_ms_hifi4:    8.5669

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
