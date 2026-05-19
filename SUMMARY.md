loader_path: third_party.tt_forge_models.unsloth_qwen2_5_1_5b.causal_lm.pytorch.loader
variant_id: Qwen2.5_1.5B
arch: p150
status: DONE_PASS
test_function: test_unsloth_qwen2_5_1_5b
samples_per_second: 68.00447522614122
ttft_ms: 148.905905
prefill_pcc: 0.992388
first_decode_pcc: 0.998932
top_perf_samples_per_sec: 206.5544
pct_of_target: 32.9
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_unsloth_qwen2_5_1_5b

## Test
tests/benchmark/test_llms.py::test_unsloth_qwen2_5_1_5b

## Model
- HF name:    unsloth/Qwen2.5-1.5B
- Loader:     third_party.tt_forge_models.unsloth_qwen2_5_1_5b.causal_lm.pytorch.loader
- Variant:    Qwen2.5_1.5B

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=2 fails with compiler error:
  ttnn.paged_update_cache failed validation — input_tensor.is_sharded() assertion in
  paged_update_cache_device_operation.cpp:160. Using optimization_level=1 as stable setting.

Also fixed general infrastructure bug in tests/benchmark/benchmarks/llm_benchmark.py:
  get_weight_dtype_config_path() was called without checking if the method exists on the loader.
  Fixed with getattr fallback so loaders without this method work correctly.

## Measured (full model, defaults)
- Sample per second:  68.00
- TTFT (ms):          148.91
- Prefill PCC:        0.992388
- First decode PCC:   0.998932
- Wall clock:         0:01:56
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_unsloth_qwen2_5_1_5b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 32.9% (68.00 / 206.55)

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
- total_flops:             98790277248
- breakdown.matmul:        93151297664
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
- count:                  1777088195
- effective_count:        1543714499
- memory_bytes:           2107080456
- memory_gb:              1.9623715952038765
- effective_memory_bytes: 1640333064
- effective_memory_gb:    1.5276792123913765
- embedding_count:        233373696
- embedding_memory_bytes: 466747392

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 206.5544
- top_perf_time_ms:         4.8413
- dram_time_ms:             3.2276
- compute_time_ms_lofi:     0.1123
- compute_time_ms_hifi2:    0.2245
- compute_time_ms_hifi3:    0.3368
- compute_time_ms_hifi4:    0.4490

## Files changed
- tests/benchmark/test_llms.py (added test_unsloth_qwen2_5_1_5b)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: getattr fallback for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
