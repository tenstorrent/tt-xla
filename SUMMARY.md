loader_path: third_party.tt_forge_models.agentdog_qwen2_5_gguf.causal_lm.pytorch.loader
variant_id: 7B_i1
arch: p150
status: DONE_PASS
test_function: test_agentdog_qwen2_5_gguf_7b_i1
samples_per_second: 5.604796509442791
ttft_ms: 668.344375
prefill_pcc: 0.989882
first_decode_pcc: 0.996436
top_perf_samples_per_sec: 46.0472
pct_of_target: 12.2
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_agentdog_qwen2_5_gguf_7b_i1

## Test
tests/benchmark/test_llms.py::test_agentdog_qwen2_5_gguf_7b_i1

## Model
- HF name:    mradermacher/AgentDoG-Qwen2.5-7B-i1-GGUF
- Loader:     third_party.tt_forge_models.agentdog_qwen2_5_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.AGENTDOG_QWEN2_5_7B_I1 ("7B_i1")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  5.604796509442791
- TTFT (ms):          668.344375
- Prefill PCC:        0.989882
- First decode PCC:   0.996436
- Wall clock:         0:42:32
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_agentdog_qwen2_5_gguf_7b_i1_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 12.2% (5.604796 / 46.0472)

Note: p150 (blackhole) shows lower efficiency (12.2%) vs n150 (73.3%). This is expected
given the relative maturity of compiler optimizations for the blackhole architecture.

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
- compute_time_ms_lofi:     0.4367
- compute_time_ms_hifi2:    0.8734
- compute_time_ms_hifi3:    1.3100
- compute_time_ms_hifi4:    1.7467

## Files changed
- tests/benchmark/test_llms.py (added test_agentdog_qwen2_5_gguf_7b_i1)
- tests/benchmark/benchmarks/llm_benchmark.py (added hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added agentdog_qwen2_5_gguf_7b_i1 entry)
- SUMMARY.md

## tt-forge-models submodule
no change
