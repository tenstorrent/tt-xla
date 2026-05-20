loader_path: third_party.tt_forge_models.meditron.causal_lm.pytorch.loader
variant_id: Meditron_7B
arch: p150
status: DONE_PASS
test_function: test_meditron_7b
samples_per_second: 21.155794847
ttft_ms: 405.474948
prefill_pcc: 0.998628
first_decode_pcc: 0.974057
top_perf_samples_per_sec: 43.0912
pct_of_target: 49.1
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_meditron_7b

## Test
tests/benchmark/test_llms.py::test_meditron_7b

## Model
- HF name:    malhajar/meditron-7b-chat
- Loader:     third_party.tt_forge_models.meditron.causal_lm.pytorch.loader
- Variant:    MEDITRON_7B

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=2 causes first decode PCC to drop to ~0.904 (below 0.94 threshold).
optimization_level=1 passes with decode PCC=0.974. experimental_weight_dtype="bfp_bf8" is the
default and was kept (removing it worsened PCC to 0.894). Also fixed a general benchmarking
infrastructure issue: llm_benchmark.py was calling get_weight_dtype_config_path() without
checking hasattr(), causing AttributeError for loaders that don't implement this optional method.

## Measured (full model, defaults)
- Sample per second:  21.155794847
- TTFT (ms):          405.474948
- Prefill PCC:        0.998628
- First decode PCC:   0.974057
- Wall clock:         0:03:12
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tests/benchmark/tt_xla_meditron_7b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 49.1%

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
- total_flops:             422857408640
- breakdown.matmul:        422857408640
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
- count:                  6738555075
- effective_count:        6607413443
- memory_bytes:           7282910216
- memory_gb:              6.782738693058491
- effective_memory_bytes: 7020626952
- effective_memory_gb:    6.5384683683514595
- embedding_count:        131141632
- embedding_memory_bytes: 262283264

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 43.0912
- top_perf_time_ms:         23.2066
- dram_time_ms:             15.4711
- compute_time_ms_lofi:     0.4805
- compute_time_ms_hifi2:    0.9610
- compute_time_ms_hifi3:    1.4416
- compute_time_ms_hifi4:    1.9221

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
