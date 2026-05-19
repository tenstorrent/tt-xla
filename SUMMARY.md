loader_path: third_party.tt_forge_models.deepseek_r1_distill_qwen_1_5b_gguf.causal_lm.pytorch.loader
variant_id: DeepSeek_R1_Distill_Qwen_1_5B_Q4_0
arch: p150
status: DONE_PASS
test_function: test_deepseek_r1_distill_qwen_1_5b_gguf
samples_per_second: 58.99
ttft_ms: 258.67
prefill_pcc: 0.9474
first_decode_pcc: 0.9631
top_perf_samples_per_sec: 206.55
pct_of_target: 28.6
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_deepseek_r1_distill_qwen_1_5b_gguf

## Test
tests/benchmark/test_llms.py::test_deepseek_r1_distill_qwen_1_5b_gguf

## Model
- HF name:    ggml-org/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0-GGUF
- Loader:     third_party.tt_forge_models.deepseek_r1_distill_qwen_1_5b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.DEEPSEEK_R1_DISTILL_QWEN_1_5B_Q4_0

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=2 fails with `ttnn.paged_update_cache` requiring sharded input;
optimization_level=1 (DRAM-only) passes.

## Measured (full model, defaults)
- Sample per second:  58.99
- TTFT (ms):          258.67
- Prefill PCC:        0.9474
- First decode PCC:   0.9631
- Wall clock:         0:05:06
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_deepseek_r1_distill_qwen_1_5b_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 28.6% (58.99 / 206.55)

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
- memory_bytes:           2107080456
- memory_gb:              1.9623715952038765
- effective_memory_bytes: 1640333064
- effective_memory_gb:    1.5276792123913765
- embedding_count:        233373696
- embedding_memory_bytes: 466747392

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 206.5544
- top_perf_time_ms:         4.8413
- dram_time_ms:             3.2276
- compute_time_ms_lofi:     0.0950
- compute_time_ms_hifi2:    0.1900
- compute_time_ms_hifi3:    0.2850
- compute_time_ms_hifi4:    0.3800

## Files changed
- tests/benchmark/test_llms.py (added test_deepseek_r1_distill_qwen_1_5b_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed: added hasattr check before get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added deepseek_r1_distill_qwen_1_5b_gguf entry)

## tt-forge-models submodule
no change
