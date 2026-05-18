loader_path: third_party.tt_forge_models.mradermacher_es_qwen_math_base_7b_3k_stage2_6k_t4_ds_o2_aug_kl0_01_step480_i1_gguf.causal_lm.pytorch.loader
variant_id: es_qwen_math_base_7B_stage2_step480_i1_GGUF
arch: p150
status: DONE_FAIL
test_function: test_mradermacher_es_qwen_math_base_7b_stage2_step480_i1_gguf
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: 46.0472
pct_of_target: null
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: null
failure_reason: "full-model PCC below required 0.94 across all optimization levels (0/1/2) and weight dtype configs (bfp_bf8, none); best prefill PCC=0.921 at OL=2/bfp_bf8, decode PCC=0.884 at OL=0; likely GGUF i1 Q4_K_M dequantization numerical accuracy limitation on TT hardware"

# Benchmark added: test_mradermacher_es_qwen_math_base_7b_stage2_step480_i1_gguf

## Test
tests/benchmark/test_llms.py::test_mradermacher_es_qwen_math_base_7b_stage2_step480_i1_gguf

## Model
- HF name:    mradermacher/es-qwen-math-base-7b-3k-stage2-6k-t4-ds_o2_aug-kl0.01-step480-i1-GGUF
- Loader:     third_party.tt_forge_models.mradermacher_es_qwen_math_base_7b_3k_stage2_6k_t4_ds_o2_aug_kl0_01_step480_i1_gguf.causal_lm.pytorch.loader
- Variant:    es_qwen_math_base_7B_stage2_step480_i1_GGUF (enum key: MRADERMACHER_ES_QWEN_MATH_BASE_7B_I1_GGUF)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: none (bfp_bf8 causes prefill PCC regression; no override passes)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  null (test failed PCC assertion before completion)
- TTFT (ms):          null
- Prefill PCC:        null (best observed: 0.921 at OL=2/bfp_bf8; fails 0.94 threshold)
- First decode PCC:   null (best observed: 0.884 at OL=0; fails 0.94 threshold)
- Wall clock:         ~8:23 (OL=2 run)
- Hardware:           p150

## Investigation summary
All combinations of optimization_level (0, 1, 2), experimental_weight_dtype (bfp_bf8, ""),
and fp32_dest_acc_en (True) were tested. None achieved PCC >= 0.94 on the full 28-layer model.
With --num-layers 1, PCC passes easily (prefill=0.970, decode=0.995 at OL=2/bfp_bf8).
The degradation with full depth strongly suggests accumulated numerical error from the
GGUF i1 Q4_K_M dequantization + TT bfloat16 arithmetic across 28 transformer layers.
Infrastructure fix applied: llm_benchmark.py get_weight_dtype_config_path() now guarded
with hasattr() to support loaders that don't implement this optional method.

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_mradermacher_es_qwen_math_base_7b_stage2_step480_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: N/A (DONE_FAIL)

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
- memory_gb:              14.185
- effective_memory_bytes: 14141239060
- effective_memory_gb:    13.170
- embedding_count:        544997376
- embedding_memory_bytes: 1089994752

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 46.0472
- top_perf_time_ms:         21.7169
- dram_time_ms:             14.4779
- compute_time_ms_lofi:     0.5161
- compute_time_ms_hifi2:    1.0322
- compute_time_ms_hifi3:    1.5482
- compute_time_ms_hifi4:    2.0643

## Files changed
- tests/benchmark/test_llms.py (added test_mradermacher_es_qwen_math_base_7b_stage2_step480_i1_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
