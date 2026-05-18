loader_path: third_party.tt_forge_models.deepseek_r1_distill_qwen_7b_gguf.causal_lm.pytorch.loader
variant_id: Distill_Qwen_7B_GGUF
arch: p150
status: DONE_FAIL
test_function: test_deepseek_r1_distill_qwen_7b_gguf
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: 46.0472
pct_of_target: null
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "Prefill PCC fails across all optimization levels (best: 0.934392 at optimization_level=0, no experimental_weight_dtype); GGUF Q4_K_M quantization error accumulates across 28 layers vs float32 CPU reference; no configuration achieves required PCC >= 0.94"

# Benchmark added: test_deepseek_r1_distill_qwen_7b_gguf

## Test
tests/benchmark/test_llms.py::test_deepseek_r1_distill_qwen_7b_gguf

## Model
- HF name:    bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF
- Loader:     third_party.tt_forge_models.deepseek_r1_distill_qwen_7b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.DISTILL_QWEN_7B_GGUF

## Test config landed
- optimization_level:        2 (DEFAULT_OPTIMIZATION_LEVEL)
- trace_enabled:             true (default)
- experimental_weight_dtype: bfp_bf8 (default)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (PCC failure)
- TTFT (ms):          N/A (PCC failure)
- Prefill PCC:        0.859371 (at default settings, opt_level=2, bfp_bf8)
- First decode PCC:   N/A (prefill assertion fires first)
- Wall clock:         N/A
- Hardware:           p150

## PCC investigation
Tried all optimization levels and weight dtype combinations. None achieve PCC >= 0.94:
- opt_level=2, bfp_bf8:          prefill PCC=0.859371
- opt_level=1, bfp_bf8:          prefill PCC=0.854957
- opt_level=1, no weight dtype:  prefill PCC=0.897431
- opt_level=0, no weight dtype:  prefill PCC=0.934392 (best achieved)
- opt_level=2, no weight dtype, fp32_dest_acc_en=True: prefill PCC=0.925671

Root cause: GGUF Q4_K_M quantization causes error accumulation across all 28 layers
when comparing bf16 device computation against float32 CPU reference. 1-layer tests
pass PCC (0.976) but error accumulates beyond threshold at full depth.

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_deepseek_r1_distill_qwen_7b_gguf_perf_metrics_3.json
Achieved vs top_perf_samples_per_sec: N/A (test did not pass PCC)

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
- memory_bytes:           15231233800
- memory_gb:              14.185191877186298
- effective_memory_bytes: 14141239048
- effective_memory_gb:    13.170055158436298
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
- tests/benchmark/test_llms.py (added test_deepseek_r1_distill_qwen_7b_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: hasattr check for get_weight_dtype_config_path)

## tt-forge-models submodule
no change (submodule at a1e7a6082a)
