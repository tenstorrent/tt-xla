loader_path: third_party.tt_forge_models.alpaca.causal_lm.pytorch.loader
variant_id: native
arch: p150
status: DONE_PASS
test_function: test_alpaca_native
samples_per_second: 25.7076
ttft_ms: 343.566849
prefill_pcc: 0.996989
first_decode_pcc: 0.957912
top_perf_samples_per_sec: 43.0915
pct_of_target: 59.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_alpaca_native

## Test
tests/benchmark/test_llms.py::test_alpaca_native

## Model
- HF name:    maicomputer/alpaca-native
- Loader:     third_party.tt_forge_models.alpaca.causal_lm.pytorch.loader
- Variant:    ModelVariant.ALPACA_NATIVE (native)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  25.7076
- TTFT (ms):          343.566849
- Prefill PCC:        0.996989
- First decode PCC:   0.957912
- Wall clock:         0:08:32
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_alpaca_native_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 59.7%

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
- total_flops:             422853214336
- breakdown.matmul:        422853214336
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
- count:                  6738424003
- effective_count:        6607347907
- memory_bytes:           7282709512
- memory_gb:              6.782551772892475
- effective_memory_bytes: 7020557320
- effective_memory_gb:    6.538403518497944
- embedding_count:        131076096
- embedding_memory_bytes: 262152192

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 43.0915
- top_perf_time_ms:         23.2064
- dram_time_ms:             15.4709
- compute_time_ms_lofi:     0.4805
- compute_time_ms_hifi2:    0.9610
- compute_time_ms_hifi3:    1.4415
- compute_time_ms_hifi4:    1.9221

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py

## tt-forge-models submodule
no change
