loader_path: third_party.tt_forge_models.bella_tao_merged_qwen2_5_coder_7b_i1_gguf.causal_lm.pytorch.loader
variant_id: 7B_i1_GGUF
arch: n150
status: DONE_PASS
test_function: test_bella_tao_merged_qwen2_5_coder_7b_i1_gguf
samples_per_second: 18.92478623642331
ttft_ms: 609.166171
prefill_pcc: 0.994802
first_decode_pcc: 0.998708
top_perf_samples_per_sec: 25.9015
pct_of_target: 73.1
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_bella_tao_merged_qwen2_5_coder_7b_i1_gguf

## Test
tests/benchmark/test_llms.py::test_bella_tao_merged_qwen2_5_coder_7b_i1_gguf

## Model
- HF name:    mradermacher/bella-tao-merged-qwen2_5-coder-7b-i1-GGUF
- Loader:     third_party.tt_forge_models.bella_tao_merged_qwen2_5_coder_7b_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.BELLA_TAO_MERGED_QWEN2_5_CODER_7B_I1_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  18.92478623642331
- TTFT (ms):          609.166171
- Prefill PCC:        0.994802
- First decode PCC:   0.998708
- Wall clock:         0:15:46
- Hardware:           n300 (wormhole_b0, single-chip)

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_bella_tao_merged_qwen2_5_coder_7b_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 73.1%

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
- top_perf_samples_per_sec: 25.9015
- top_perf_time_ms:         38.6077
- dram_time_ms:             25.7385
- compute_time_ms_lofi:     1.7676
- compute_time_ms_hifi2:    3.5352
- compute_time_ms_hifi3:    5.3028
- compute_time_ms_hifi4:    7.0704

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py

## tt-forge-models submodule
no change
