loader_path: third_party.tt_forge_models.cookgptlama.causal_lm.pytorch.loader
variant_id: cookgptlama
arch: n150
status: DONE_PASS
test_function: test_cookgptlama
samples_per_second: 59.52
ttft_ms: 332.98
prefill_pcc: 0.997187
first_decode_pcc: 0.998167
top_perf_samples_per_sec: 172.0573
pct_of_target: 34.6
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_cookgptlama

## Test
tests/benchmark/test_llms.py::test_cookgptlama

## Model
- HF name:    VishalMysore/cookgptlama
- Loader:     third_party.tt_forge_models.cookgptlama.causal_lm.pytorch.loader
- Variant:    ModelVariant.COOKGPTLAMA

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  59.52
- TTFT (ms):          332.98
- Prefill PCC:        0.997187
- First decode PCC:   0.998167
- Wall clock:         0:07:22
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_cookgptlama_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 34.6% (59.52 / 172.06)

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
- total_flops:             66295169088
- breakdown.matmul:        66295169088
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
- count:                  1101490340
- effective_count:        1035954340
- memory_bytes:           1231860362
- memory_gb:              1.1472593639045954
- effective_memory_bytes: 1100788362
- effective_memory_gb:    1.0251890514045954
- embedding_count:        65536000
- embedding_memory_bytes: 131072000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 172.0573
- top_perf_time_ms:         5.8120
- dram_time_ms:             3.8747
- compute_time_ms_lofi:     0.2590
- compute_time_ms_hifi2:    0.5179
- compute_time_ms_hifi3:    0.7769
- compute_time_ms_hifi4:    1.0359

## Files changed
- tests/benchmark/test_llms.py (added test_cookgptlama)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added cookgptlama entry)

## tt-forge-models submodule
no change
