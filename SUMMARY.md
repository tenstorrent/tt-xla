loader_path: third_party.tt_forge_models.deepseek_r1_distill_qwen_7b_uncensored_i1_gguf.causal_lm.pytorch.loader
variant_id: 7B_Uncensored_i1_GGUF
arch: n150
status: DONE_FAIL
test_function: test_deepseek_r1_distill_qwen_7b_uncensored_i1_gguf
samples_per_second: 19.25
ttft_ms: 609.8
prefill_pcc: 0.827
first_decode_pcc: null
top_perf_samples_per_sec: 25.9067
pct_of_target: 74.3
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "GGUF Q4_K_M model double-quantized to bfp_bf8 produces insufficient prefill PCC (best 0.827 at optimization_level=2, required 0.94); without bfp_bf8 compilation fails with INTERNAL: Error code: 13 (OOM)"

# Benchmark added: test_deepseek_r1_distill_qwen_7b_uncensored_i1_gguf

## Test
tests/benchmark/test_llms.py::test_deepseek_r1_distill_qwen_7b_uncensored_i1_gguf

## Model
- HF name:    mradermacher/DeepSeek-R1-Distill-Qwen-7B-Uncensored-i1-GGUF
- Loader:     third_party.tt_forge_models.deepseek_r1_distill_qwen_7b_uncensored_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.DEEPSEEK_R1_DISTILL_QWEN_7B_UNCENSORED_I1_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  19.25 (from failing run — PCC check failed)
- TTFT (ms):          609.8 (from failing run)
- Prefill PCC:        0.827 (failed, required 0.94)
- First decode PCC:   N/A (test aborted at prefill PCC check)
- Wall clock:         0:15:57
- Hardware:           n150 (wormhole_b0, n300 chassis)

## Failure Analysis
The GGUF Q4_K_M model is loaded via transformers AutoModelForCausalLM which
dequantizes the 4-bit GGUF weights to bf16. The TT device then re-applies
bfp_bf8 quantization via `experimental_weight_dtype`. This double quantization
causes cumulative error across all 28 transformer layers, resulting in
insufficient prefill PCC:

| optimization_level | experimental_weight_dtype | Prefill PCC | Status     |
|--------------------|--------------------------|-------------|------------|
| 2                  | bfp_bf8 (default)        | 0.827       | FAIL (<0.94) |
| 1                  | bfp_bf8 (default)        | 0.740       | FAIL (<0.94) |
| 0                  | bfp_bf8 (default)        | 0.813       | FAIL (<0.94) |
| 2                  | "" (disabled)            | N/A         | INTERNAL: Error 13 (OOM) |
| 1                  | "" (disabled)            | N/A         | INTERNAL: Error 13 (OOM) |

Single-layer sanity check: 1-layer + bfp_bf8 passes PCC (0.977), confirming
the issue is cumulative quantization error across all 28 layers, not a
per-layer compile bug.

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_deepseek_r1_distill_qwen_7b_uncensored_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 74.3% (19.25 / 25.91) — from failing run

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
- total_flops:             454055067776
- breakdown.matmul:        424455929984
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
- count:                  7612756678
- effective_count:        7069189318
- memory_bytes:           8598461428
- memory_gb:              8.007941234856844
- effective_memory_bytes: 7511326708
- effective_memory_gb:    6.995468128472567
- embedding_count:        543567360
- embedding_memory_bytes: 1087134720

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 25.9067
- top_perf_time_ms:         38.6001
- dram_time_ms:             25.7334
- compute_time_ms_lofi:     1.7737
- compute_time_ms_hifi2:    3.5473
- compute_time_ms_hifi3:    5.3210
- compute_time_ms_hifi4:    7.0946

## Files changed
- tests/benchmark/test_llms.py (added test_deepseek_r1_distill_qwen_7b_uncensored_i1_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (added hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added deepseek_r1_distill_qwen_7b_uncensored_i1_gguf entry)

## tt-forge-models submodule
no change
