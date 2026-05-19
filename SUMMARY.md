loader_path: third_party.tt_forge_models.mmnga_deepseek_r1_distill_qwen_14b_gguf.causal_lm.pytorch.loader
variant_id: mmnga_DeepSeek_R1_Distill_Qwen_14B_gguf
arch: p150
status: DONE_PASS
test_function: test_mmnga_deepseek_r1_distill_qwen_14b_gguf
samples_per_second: 15.886
ttft_ms: 547.347
prefill_pcc: 0.997859
first_decode_pcc: 0.997911
top_perf_samples_per_sec: 22.9948
pct_of_target: 69.1
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_mmnga_deepseek_r1_distill_qwen_14b_gguf

## Test
tests/benchmark/test_llms.py::test_mmnga_deepseek_r1_distill_qwen_14b_gguf

## Model
- HF name:    mmnga/DeepSeek-R1-Distill-Qwen-14B-gguf
- Loader:     third_party.tt_forge_models.mmnga_deepseek_r1_distill_qwen_14b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.MMNGA_DEEPSEEK_R1_DISTILL_QWEN_14B_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  15.886
- TTFT (ms):          547.347
- Prefill PCC:        0.997859
- First decode PCC:   0.997911
- Wall clock:         0:18:07
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_mmnga_deepseek_r1_distill_qwen_14b_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 69.1% (15.886 / 22.9948)

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
- total_flops:             895411028096
- breakdown.matmul:        782657126528
- breakdown.linear:        112753901568
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        402653184
- memory_bytes: 805306368
- memory_gb:    0.75

### Params
- count:                  14770033859
- effective_count:        13991466179
- memory_bytes:           16423856904
- memory_gb:              15.30
- effective_memory_bytes: 14866721544
- effective_memory_gb:    13.85
- embedding_count:        778567680
- embedding_memory_bytes: 1557135360

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 22.9948
- top_perf_time_ms:         43.4881
- dram_time_ms:             28.9921
- compute_time_ms_lofi:     1.0175
- compute_time_ms_hifi2:    2.0350
- compute_time_ms_hifi3:    3.0525
- compute_time_ms_hifi4:    4.0701

## Files changed
- tests/benchmark/test_llms.py (added test_mmnga_deepseek_r1_distill_qwen_14b_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed get_weight_dtype_config_path hasattr guard)

## tt-forge-models submodule
no change
