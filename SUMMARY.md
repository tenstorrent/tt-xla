loader_path: third_party.tt_forge_models.lmstudio_qwen_2_5_coder_1_5b_instruct_gguf.causal_lm.pytorch.loader
variant_id: 1.5B_Instruct_GGUF
arch: p150
status: DONE_PASS
test_function: test_lmstudio_qwen_2_5_coder_1_5b_instruct_gguf
samples_per_second: 67.51
ttft_ms: 150.60
prefill_pcc: 0.996454
first_decode_pcc: 0.998736
top_perf_samples_per_sec: 206.5544
pct_of_target: 32.7
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_lmstudio_qwen_2_5_coder_1_5b_instruct_gguf

## Test
tests/benchmark/test_llms.py::test_lmstudio_qwen_2_5_coder_1_5b_instruct_gguf

## Model
- HF name:    lmstudio-community/Qwen2.5-Coder-1.5B-Instruct-GGUF
- Loader:     third_party.tt_forge_models.lmstudio_qwen_2_5_coder_1_5b_instruct_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.QWEN_2_5_CODER_1_5B_INSTRUCT_GGUF ("1.5B_Instruct_GGUF")

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Notes
- optimization_level=2 fails with: `paged_update_cache` expects input_tensor to be sharded
  (OperationValidationAndFallback in paged_update_cache_device_operation.cpp:160)
- optimization_level=1 passes cleanly on full model

## Infrastructure fix
- tests/benchmark/benchmarks/llm_benchmark.py: added `hasattr` guard around
  `model_loader.get_weight_dtype_config_path()` call; not all ForgeModel loaders
  implement this method, and the harness should degrade gracefully to no-op
  rather than raise AttributeError.

## Measured (full model, defaults)
- Sample per second:  67.51
- TTFT (ms):          150.60
- Prefill PCC:        0.996454
- First decode PCC:   0.998736
- Wall clock:         0:02:24
- Hardware:           p150 (blackhole, single chip of p300c board)

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_lmstudio_qwen_2_5_coder_1_5b_instruct_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 32.7% (67.51 / 206.55)
Note: gap from roofline is due to optimization_level=1 (no SRAM), not a correctness issue.

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
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py

## tt-forge-models submodule
no change
