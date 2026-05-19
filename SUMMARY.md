loader_path: third_party.tt_forge_models.quantfactory_qwen_2_5_sex_gguf.causal_lm.pytorch.loader
variant_id: QUANTFACTORY_QWEN_2_5_SEX_GGUF
arch: p150
status: DONE_PASS
test_function: test_quantfactory_qwen_2_5_sex_gguf
samples_per_second: 59.98
ttft_ms: 153.63
prefill_pcc: 0.992397
first_decode_pcc: 0.997340
top_perf_samples_per_sec: 206.5544
pct_of_target: 29.0
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: null
failure_reason: null

# Benchmark added: test_quantfactory_qwen_2_5_sex_gguf

## Test
tests/benchmark/test_llms.py::test_quantfactory_qwen_2_5_sex_gguf

## Model
- HF name:    QuantFactory/Qwen2.5-Sex-GGUF
- Loader:     third_party.tt_forge_models.quantfactory_qwen_2_5_sex_gguf.causal_lm.pytorch.loader
- Variant:    QUANTFACTORY_QWEN_2_5_SEX_GGUF

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: none (disabled: GGUF Q4_K_M weights + bfp_bf8 causes decode PCC drop to ~0.82)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Infrastructure fix
Fixed `tests/benchmark/benchmarks/llm_benchmark.py` to guard `get_weight_dtype_config_path()`
with `hasattr` before calling it, matching the pattern already used in the runner
(`tests/runner/testers/torch/dynamic_torch_model_tester.py`). Without this fix, any
loader that doesn't implement `get_weight_dtype_config_path()` (including all new
GGUF loaders) raises `AttributeError` before the benchmark starts.

## Measured (full model, defaults)
- Sample per second:  59.98
- TTFT (ms):          153.63
- Prefill PCC:        0.992397
- First decode PCC:   0.997340
- Wall clock:         0:02:15
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_quantfactory_qwen_2_5_sex_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 29.0% (59.98 / 206.55)

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
- tests/benchmark/test_llms.py (added test_quantfactory_qwen_2_5_sex_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json (added quantfactory_qwen_2_5_sex_gguf entry)

## tt-forge-models submodule
no change
