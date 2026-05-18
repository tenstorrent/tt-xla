loader_path: third_party.tt_forge_models.deepseek_r1_distill_qwen_7b_gspo_basic_i1_gguf.causal_lm.pytorch.loader
variant_id: DISTILL_QWEN_7B_GSPO_BASIC_I1_Q4_K_M_GGUF
arch: p150
status: DONE_FAIL
test_function: test_deepseek_r1_distill_qwen_7b_gspo_basic_i1_gguf
samples_per_second: 37.82
ttft_ms: 243.88
prefill_pcc: 0.851137
first_decode_pcc: null
top_perf_samples_per_sec: 46.0472
pct_of_target: 82.1
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "Prefill PCC=0.851 below required 0.94 with all configuration variants tested (opt=2+bfp_bf8, opt=2+no-bfp_bf8, opt=1+no-bfp_bf8, opt=2+bfp_bf8+fp32_dest_acc_en); GGUF Q4_K_M quantization causes accumulated numerical error over 28 layers"

# Benchmark added: test_deepseek_r1_distill_qwen_7b_gspo_basic_i1_gguf

## Test
tests/benchmark/test_llms.py::test_deepseek_r1_distill_qwen_7b_gspo_basic_i1_gguf

## Model
- HF name:    mradermacher/DeepSeek-R1-Distill-Qwen-7B-GSPO-Basic-i1-GGUF
- Loader:     third_party.tt_forge_models.deepseek_r1_distill_qwen_7b_gspo_basic_i1_gguf.causal_lm.pytorch.loader
- Variant:    DISTILL_QWEN_7B_GSPO_BASIC_I1_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8 (default)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  37.82 (measured, but test fails PCC)
- TTFT (ms):          243.88
- Prefill PCC:        0.851137 (FAIL — required 0.94)
- First decode PCC:   null (not reached due to prefill PCC failure)
- Wall clock:         0:08:24
- Hardware:           p150

## PCC Investigation
All configurations tried for the full 28-layer model:
- opt=2 + bfp_bf8 (default):               Prefill PCC=0.851 (FAIL)
- opt=2 + bfp_bf8 + fp32_dest_acc_en=True: Prefill PCC=0.851 (FAIL)
- opt=2 + no bfp_bf8 (experimental_weight_dtype=""): Prefill PCC=0.792 (FAIL)
- opt=1 + no bfp_bf8:                      Prefill PCC=0.733 (FAIL)

At 1 layer, all configs pass (PCC ~0.977-0.986), confirming the
failure is accumulated numerical error across the 28 layers of the
GGUF Q4_K_M quantized model.

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_deepseek_r1_distill_qwen_7b_gspo_basic_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 37.82 / 46.05 = 82.1%

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
- total_flops:             452502421632
- breakdown.matmul:        422903283840
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
- count:                  7615616707
- effective_count:        7070619331
- memory_bytes:           8602840840
- memory_gb:              8.012019880115986
- effective_memory_bytes: 7512846088
- effective_memory_gb:    6.996883161365986
- embedding_count:        544997376
- embedding_memory_bytes: 1089994752

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 46.0472
- top_perf_time_ms:         21.7169
- dram_time_ms:             14.4779
- compute_time_ms_lofi:     0.5142
- compute_time_ms_hifi2:    1.0284
- compute_time_ms_hifi3:    1.5426
- compute_time_ms_hifi4:    2.0568

## Files changed
- tests/benchmark/test_llms.py (new test function)
- tests/benchmark/benchmarks/llm_benchmark.py (hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
