loader_path: third_party.tt_forge_models.deepseek_r1_distill_qwen_7b_uncensored_i1_gguf.causal_lm.pytorch.loader
variant_id: 7B_Uncensored_i1_GGUF
arch: p150
status: DONE_FAIL
test_function: test_deepseek_r1_distill_qwen_7b_uncensored_i1_gguf
samples_per_second: 37.798511766549744
ttft_ms: 270.003828
prefill_pcc: 0.872542
first_decode_pcc: null
top_perf_samples_per_sec: 46.0563
pct_of_target: 82.1
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "Prefill PCC 0.872542 < required 0.94 across all tested configurations (bfp_bf8+opt_level=2: 0.872, no-bfp_bf8+opt_level=2: 0.745, no-bfp_bf8+opt_level=1: 0.842, no-bfp_bf8+opt_level=0+fp32_dest_acc_en=False: 0.582); GGUF Q4_K_M quantization error accumulates across 28 layers causing TT bfloat16 vs CPU float32 reference divergence"

# Benchmark added: test_deepseek_r1_distill_qwen_7b_uncensored_i1_gguf

## Test
tests/benchmark/test_llms.py::test_deepseek_r1_distill_qwen_7b_uncensored_i1_gguf

## Model
- HF name:    mradermacher/DeepSeek-R1-Distill-Qwen-7B-Uncensored-i1-GGUF
- Loader:     third_party.tt_forge_models.deepseek_r1_distill_qwen_7b_uncensored_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.DEEPSEEK_R1_DISTILL_QWEN_7B_UNCENSORED_I1_GGUF (= "7B_Uncensored_i1_GGUF")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8 (default)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults — PCC FAILED)
- Sample per second:  37.798511766549744 (bfp_bf8 + opt_level=2 run; perf ran but PCC failed)
- TTFT (ms):          270.003828
- Prefill PCC:        0.872542 (best observed, with bfp_bf8 + opt_level=2)
- First decode PCC:   null (not reached; prefill assertion fires first)
- Wall clock:         0:08:24
- Hardware:           p150

## PCC Investigation
All configurations tested — none reached the 0.94 threshold:

| Configuration                              | Prefill PCC |
|--------------------------------------------|-------------|
| bfp_bf8 + opt_level=2 (default)           | 0.872542    |
| no bfp_bf8 + opt_level=2                  | 0.745323    |
| no bfp_bf8 + opt_level=1                  | 0.842756    |
| no bfp_bf8 + opt_level=0 + fp32_dest_acc_en=False | 0.582533 |

Root cause: GGUF Q4_K_M quantization error accumulates across 28 model layers, causing divergence between TT bfloat16 computation and CPU float32 reference. Single-layer tests pass cleanly (Prefill PCC ~0.971–0.977, Decode PCC ~0.986–0.988), confirming the error is additive per layer.

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_deepseek_r1_distill_qwen_7b_uncensored_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 82.1% (37.8 / 46.1 sps)

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
- memory_bytes:           15225513748
- memory_gb:              14.179864663630724
- effective_memory_bytes: 14138379028
- effective_memory_gb:    13.167391557246447
- embedding_count:        543567360
- embedding_memory_bytes: 1087134720

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 46.0563
- top_perf_time_ms:         21.7125
- dram_time_ms:             14.4750
- compute_time_ms_lofi:     0.5160
- compute_time_ms_hifi2:    1.0319
- compute_time_ms_hifi3:    1.5479
- compute_time_ms_hifi4:    2.0639

## Files changed
- tests/benchmark/test_llms.py (added test_deepseek_r1_distill_qwen_7b_uncensored_i1_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed get_weight_dtype_config_path AttributeError for loaders without the method)

## tt-forge-models submodule
no change
