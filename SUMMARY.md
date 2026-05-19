loader_path: third_party.tt_forge_models.richarderkhov_ehristoforu_qwen2_1_5b_it_chat_gguf.causal_lm.pytorch.loader
variant_id: 1.5b_it_chat_GGUF
arch: p150
status: DONE_PASS
test_function: test_richarderkhov_ehristoforu_qwen2_1_5b_it_chat_gguf
samples_per_second: 67.74
ttft_ms: 147.96
prefill_pcc: 0.990127
first_decode_pcc: 0.992022
top_perf_samples_per_sec: 206.5544
pct_of_target: 32.8
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_richarderkhov_ehristoforu_qwen2_1_5b_it_chat_gguf

## Test
tests/benchmark/test_llms.py::test_richarderkhov_ehristoforu_qwen2_1_5b_it_chat_gguf

## Model
- HF name:    RichardErkhov/ehristoforu_-_Qwen2-1.5b-it-chat-gguf
- Loader:     third_party.tt_forge_models.richarderkhov_ehristoforu_qwen2_1_5b_it_chat_gguf.causal_lm.pytorch.loader
- Variant:    RICHARDERKHOV_EHRISTOFORU_QWEN2_1_5B_IT_CHAT_GGUF = "1.5b_it_chat_GGUF"

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  67.74
- TTFT (ms):          147.96
- Prefill PCC:        0.990127
- First decode PCC:   0.992022
- Wall clock:         0:02:24
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_richarderkhov_ehristoforu_qwen2_1_5b_it_chat_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 32.8% (67.74 / 206.55 sps)

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
- compute_time_ms_lofi:     0.1123
- compute_time_ms_hifi2:    0.2245
- compute_time_ms_hifi3:    0.3368
- compute_time_ms_hifi4:    0.4490

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py

## tt-forge-models submodule
no change
