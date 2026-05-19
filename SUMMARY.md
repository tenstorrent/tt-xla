loader_path: third_party.tt_forge_models.qwen_2_5_sex_gguf.causal_lm.pytorch.loader
variant_id: QWEN_2_5_SEX_GGUF
arch: p150
status: DONE_PASS
test_function: test_qwen_2_5_sex_gguf
samples_per_second: 58.99
ttft_ms: 156.22
prefill_pcc: 0.992397
first_decode_pcc: 0.997340
top_perf_samples_per_sec: 206.5544
pct_of_target: 28.6
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: null
failure_reason: null

# Benchmark added: qwen_2_5_sex_gguf

## Test
tests/benchmark/test_llms.py::test_qwen_2_5_sex_gguf

## Model
- HF name:    mradermacher/Qwen2.5-Sex-GGUF
- Loader:     third_party.tt_forge_models.qwen_2_5_sex_gguf.causal_lm.pytorch.loader
- Variant:    QWEN_2_5_SEX_GGUF

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: none (disabled — bfp_bf8 causes PCC failure on GGUF Q4_K_M dequantized model; decode PCC dropped to 0.823 vs required 0.94)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Notes:
- optimization_level=2 fails with compiler error: ttnn.paged_update_cache validation failure
  (input_tensor.is_sharded() assert — paged_update_cache requires sharded input at opt level 2)
- experimental_weight_dtype="" (disabled): bfp_bf8 causes first decode PCC to drop to 0.823
  on this already-quantized GGUF Q4_K_M model, below the 0.94 threshold
- Infrastructure fix: added getattr guard for get_weight_dtype_config_path in llm_benchmark.py
  (method was called unconditionally but not implemented in ForgeModel base class)

## Measured (full model, defaults)
- Sample per second:  58.99
- TTFT (ms):          156.22
- Prefill PCC:        0.992397
- First decode PCC:   0.997340
- Wall clock:         0:02:16
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_qwen_2_5_sex_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 28.6% (58.99 / 206.55)
(Gap explained by optimization_level=1 and disabled bfp_bf8)

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
- memory_bytes:           3554176776
- memory_gb:              3.310085065662861
- effective_memory_bytes: 3087429384
- effective_memory_gb:    2.875392682850361
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
- tests/benchmark/test_llms.py (added test_qwen_2_5_sex_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (getattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added qwen_2_5_sex_gguf entry)

## tt-forge-models submodule
no change
