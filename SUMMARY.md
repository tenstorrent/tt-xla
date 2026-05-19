loader_path: third_party.tt_forge_models.adasearch_qwen2_5_7b_instruct_gguf.causal_lm.pytorch.loader
variant_id: 7B_Instruct_GGUF
arch: p150
status: DONE_PASS
test_function: test_adasearch_qwen2_5_7b_instruct_gguf
samples_per_second: 4.042893565013293
ttft_ms: 797.702866
prefill_pcc: 0.987598
first_decode_pcc: 0.991624
top_perf_samples_per_sec: 46.0472
pct_of_target: 8.8
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: null
failure_reason: null

# Benchmark added: test_adasearch_qwen2_5_7b_instruct_gguf

## Test
tests/benchmark/test_llms.py::test_adasearch_qwen2_5_7b_instruct_gguf

## Model
- HF name:    mradermacher/AdaSearch-Qwen2.5-7B-Instruct-GGUF
- Loader:     third_party.tt_forge_models.adasearch_qwen2_5_7b_instruct_gguf.causal_lm.pytorch.loader
- Variant:    7B_Instruct_GGUF (ModelVariant.ADASEARCH_QWEN2_5_7B_INSTRUCT_GGUF)

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: none (disabled; GGUF model with Q4_K_M quantization causes PCC regression with bfp_bf8 double-quantization)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  4.042893565013293
- TTFT (ms):          797.702866
- Prefill PCC:        0.987598
- First decode PCC:   0.991624
- Wall clock:         0:44:55
- Hardware:           p150 (blackhole)

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_adasearch_qwen2_5_7b_instruct_gguf_perf_metrics_3.json
Achieved vs top_perf_samples_per_sec: 8.8% (4.04 / 46.05)

### System
- arch:                        blackhole
- chip_count_in_system_desc:   1
- single_chip_assumption:      True
- worker_grid_cores:           130
- dram_bandwidth_bytes_per_sec: 512000000000

### Peak FLOPs
- lofi:  1040000000000000
- hifi2: 520000000000000
- hifi3: 346666666666666
- hifi4: 260000000000000

### Compute
- total_flops:             454146588800
- breakdown.matmul:        424547451008
- breakdown.linear:        29599137792
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        117440512
- memory_bytes: 234881024
- memory_gb:    0.21875

### Params
- count:                  7615616710
- effective_count:        7070619334
- memory_bytes:           15231233812
- memory_gb:              14.18519188836217
- effective_memory_bytes: 14141239060
- effective_memory_gb:    13.17005516961217
- embedding_count:        544997376
- embedding_memory_bytes: 1089994752

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 46.0472
- top_perf_time_ms:         21.7169
- dram_time_ms:             14.4779
- compute_time_ms_lofi:     0.4367
- compute_time_ms_hifi2:    0.8734
- compute_time_ms_hifi3:    1.3100
- compute_time_ms_hifi4:    1.7467

## Files changed
- tests/benchmark/test_llms.py (added test_adasearch_qwen2_5_7b_instruct_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (fix: use hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change

## Notes
- The GGUF model (Q4_K_M quantization) causes PCC regression when bfp_bf8 is applied on
  top (double quantization: GGUF already quantized, then bfp_bf8 adds another layer).
  experimental_weight_dtype="" disables bfp_bf8 to avoid this. As a result, the measured
  throughput (4.04 samples/sec) is lower than the roofline (46.05 samples/sec) which was
  computed assuming bfp_bf8 weights.
- optimization_level=2 also causes decode PCC failure (0.876 vs 0.94 threshold), so
  optimization_level=1 is used.
- llm_benchmark.py fix: added hasattr guard before calling get_weight_dtype_config_path()
  to handle loaders that don't implement this optional method (general fix, not specific
  to this model).
