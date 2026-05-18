loader_path: third_party.tt_forge_models.deepseek_r1_distill_qwen_gguf.causal_lm.pytorch.loader
variant_id: 7B_GGUF
arch: p150
status: DONE_FAIL
test_function: test_deepseek_r1_distill_qwen_gguf_7b
samples_per_second: null
ttft_ms: null
prefill_pcc: 0.962870
first_decode_pcc: 0.891772
top_perf_samples_per_sec: 46.0472
pct_of_target: null
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "Full model PCC failure: best result at optimization_level=0 gives prefill PCC=0.963 (pass) but first decode PCC=0.892 < 0.94 required; GGUF Q4_K_M dequantization accumulates numerical error in KV cache across 28 layers; all opt-levels (0,1,2) and weight-dtype configs tried"

# Benchmark added: deepseek_r1_distill_qwen_gguf_7b

## Test
tests/benchmark/test_llms.py::test_deepseek_r1_distill_qwen_gguf_7b

## Model
- HF name:    unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF (file: DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf)
- Loader:     third_party.tt_forge_models.deepseek_r1_distill_qwen_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.DEEPSEEK_R1_DISTILL_QWEN_7B_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8 (default)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test fails PCC check before reporting throughput)
- TTFT (ms):          N/A
- Prefill PCC:        0.963 (at optimization_level=0; fails at opt_level=1,2)
- First decode PCC:   0.892 (best — at optimization_level=0; below 0.94 threshold)
- Wall clock:         ~7:52 (opt_level=0 run)
- Hardware:           p150 (blackhole)

## Tuning history
All configurations tried:
- opt_level=2, bfp_bf8 (default): prefill PCC=0.920 FAIL
- opt_level=2, no bfp_bf8:        prefill PCC=0.797 FAIL (worse)
- opt_level=1, bfp_bf8:           prefill PCC=0.802 FAIL
- opt_level=0, bfp_bf8:           prefill PCC=0.963 PASS, decode PCC=0.892 FAIL
- opt_level=0, bfp_bf8, fp32_dest_acc_en=True: same as above (0.963 / 0.892)

Root cause: The Q4_K_M GGUF weights dequantize to bfloat16 with ~4-bit approximation
error per weight. Over 28 transformer layers, this accumulates in the KV cache such
that the device's decode step (using device KV cache) diverges from the CPU reference
(using CPU KV cache) by more than the 6% PCC tolerance allows. The prefill logit PCC
can pass at opt_level=0 (0.963), but the KV cache intermediate values are less accurate
than the final logits, causing decode PCC to remain at 0.892.

## Infrastructure fix also included
Fixed `tests/benchmark/benchmarks/llm_benchmark.py`: guarded
`model_loader.get_weight_dtype_config_path()` call with `hasattr()` check to avoid
`AttributeError` on loaders that don't implement this method (such as GGUF loaders).
This matches the existing pattern in `tests/runner/testers/torch/dynamic_torch_model_tester.py`.

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_deepseek_r1_distill_qwen_gguf_7b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: N/A (test failed PCC)

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
- memory_bytes:           8602840852
- memory_gb:              8.012019891291857
- effective_memory_bytes: 7512846100
- effective_memory_gb:    6.996883172541857
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
- tests/benchmark/test_llms.py (added test_deepseek_r1_distill_qwen_gguf_7b)
- .github/workflows/perf-bench-matrix.json (added deepseek_r1_distill_qwen_gguf_7b entry)
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
