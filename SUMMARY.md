loader_path: third_party.tt_forge_models.bird_chat_gguf.causal_lm.pytorch.loader
variant_id: 7B_i1_GGUF
arch: p150
status: DONE_PASS
test_function: test_bird_chat_7b_i1_gguf
samples_per_second: 37.761
ttft_ms: 245.93
prefill_pcc: 0.990628
first_decode_pcc: 0.993972
top_perf_samples_per_sec: 46.0472
pct_of_target: 82.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_bird_chat_7b_i1_gguf

## Test
tests/benchmark/test_llms.py::test_bird_chat_7b_i1_gguf

## Model
- HF name:    mradermacher/bird-chat-7b-i1-GGUF
- Loader:     third_party.tt_forge_models.bird_chat_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.BIRD_CHAT_7B_I1_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  37.761
- TTFT (ms):          245.93
- Prefill PCC:        0.990628
- First decode PCC:   0.993972
- Wall clock:         0:08:44
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_bird_chat_7b_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 82.0%

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
- tests/benchmark/test_llms.py (added test_bird_chat_7b_i1_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json (added bird_chat_7b_i1_gguf entry)

## tt-forge-models submodule
no change
