loader_path: third_party.tt_forge_models.react_native_executorch_hammer_2_1.causal_lm.pytorch.loader
variant_id: 1_5B
arch: p150
status: DONE_PASS
test_function: test_react_native_executorch_hammer_2_1_1_5b
samples_per_second: 68.04
ttft_ms: 147.97
prefill_pcc: 0.997568
first_decode_pcc: 0.998728
top_perf_samples_per_sec: 206.608
pct_of_target: 32.9
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_react_native_executorch_hammer_2_1_1_5b

## Test
tests/benchmark/test_llms.py::test_react_native_executorch_hammer_2_1_1_5b

## Model
- HF name:    MadeAgents/Hammer2.1-1.5b
- Loader:     third_party.tt_forge_models.react_native_executorch_hammer_2_1.causal_lm.pytorch.loader
- Variant:    ModelVariant.HAMMER_2_1_1_5B

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=2 fails with compiler error: ttnn.paged_update_cache expects sharded input tensor.

## Measured (full model, defaults)
- Sample per second:  68.04
- TTFT (ms):          147.97
- Prefill PCC:        0.997568
- First decode PCC:   0.998728
- Wall clock:         0:01:57
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_react_native_executorch_hammer_2_1_1_5b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 32.9% (68.04 / 206.608)

Note: Gap from roofline is expected — optimization_level=1 (not 2) is used because optimization_level=2 triggers a compiler failure (ttnn.paged_update_cache: Expect input_tensor to be sharded).

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
- total_flops:             98763636864
- breakdown.matmul:        93124657280
- breakdown.linear:        5638979584
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        58720256
- memory_bytes: 117440512
- memory_gb:    0.109375

### Params
- count:                  1776255684
- effective_count:        1543298244
- memory_bytes:           2105805676
- memory_gb:              1.9611843638122082
- effective_memory_bytes: 1639890796
- effective_memory_gb:    1.5272673182189465
- embedding_count:        232957440
- embedding_memory_bytes: 465914880

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 206.6080
- top_perf_time_ms:         4.8401
- dram_time_ms:             3.2267
- compute_time_ms_lofi:     0.1122
- compute_time_ms_hifi2:    0.2245
- compute_time_ms_hifi3:    0.3367
- compute_time_ms_hifi4:    0.4489

## Files changed
- tests/benchmark/test_llms.py (added test_react_native_executorch_hammer_2_1_1_5b)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed missing hasattr check for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
