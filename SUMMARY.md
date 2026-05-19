loader_path: third_party.tt_forge_models.yemiao2745_qwen2_5_7b_instruct_uncensored_q4_k_m_gguf.causal_lm.pytorch.loader
variant_id: 7B_Instruct_Uncensored_Q4_K_M_GGUF
arch: p150
status: DONE_PASS
test_function: test_yemiao2745_qwen2_5_7b_instruct_uncensored_q4_k_m_gguf
samples_per_second: 28.3841708239908
ttft_ms: 305.980952
prefill_pcc: 0.985829
first_decode_pcc: 0.946625
top_perf_samples_per_sec: 46.0472
pct_of_target: 61.6
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_yemiao2745_qwen2_5_7b_instruct_uncensored_q4_k_m_gguf

## Test
tests/benchmark/test_llms.py::test_yemiao2745_qwen2_5_7b_instruct_uncensored_q4_k_m_gguf

## Model
- HF name:    yemiao2745/Qwen2.5-7B-Instruct-Uncensored-Q4_K_M-GGUF
- Loader:     third_party.tt_forge_models.yemiao2745_qwen2_5_7b_instruct_uncensored_q4_k_m_gguf.causal_lm.pytorch.loader
- Variant:    7B_Instruct_Uncensored_Q4_K_M_GGUF

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  28.38
- TTFT (ms):          305.98
- Prefill PCC:        0.985829
- First decode PCC:   0.946625
- Wall clock:         0:04:34
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_yemiao2745_qwen2_5_7b_instruct_uncensored_q4_k_m_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 61.6%

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
- tests/benchmark/test_llms.py (new test function test_yemiao2745_qwen2_5_7b_instruct_uncensored_q4_k_m_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (general harness fix: graceful fallback when loader lacks get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added yemiao2745_qwen2_5_7b_instruct_uncensored_q4_k_m_gguf entry)

## tt-forge-models submodule
no change
