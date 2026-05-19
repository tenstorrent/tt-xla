loader_path: third_party.tt_forge_models.bartowski_glm_4_9b_chat_gguf.causal_lm.pytorch.loader
variant_id: 9B_CHAT_Q4_K_M
arch: p150
status: DONE_PASS
test_function: test_bartowski_glm_4_9b_chat_gguf
samples_per_second: 13.372656903386305
ttft_ms: 1341.16554
prefill_pcc: 0.989548
first_decode_pcc: 0.987173
top_perf_samples_per_sec: 37.3434
pct_of_target: 35.8
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: bartowski_glm_4_9b_chat_gguf

## Test
tests/benchmark/test_llms.py::test_bartowski_glm_4_9b_chat_gguf

## Model
- HF name:    bartowski/glm-4-9b-chat-GGUF
- Loader:     third_party.tt_forge_models.bartowski_glm_4_9b_chat_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.BARTOWSKI_GLM_4_9B_CHAT_Q4_K_M (9B_CHAT_Q4_K_M)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  13.372656903386305
- TTFT (ms):          1341.16554
- Prefill PCC:        0.989548
- First decode PCC:   0.987173
- Wall clock:         0:33:16
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_bartowski_glm_4_9b_chat_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 35.8% (13.37 / 37.34 samples/sec)

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
- total_flops:             561841307712
- breakdown.matmul:        513517027392
- breakdown.linear:        48324280320
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        83886080
- memory_bytes: 167772160
- memory_gb:    0.15625

### Params
- count:                  9400279203
- effective_count:        8779522211
- memory_bytes:           10570547848
- memory_gb:              9.844589836895466
- effective_memory_bytes: 9329033864
- effective_memory_gb:    8.688339836895466
- embedding_count:        620756992
- embedding_memory_bytes: 1241513984

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 37.3434
- top_perf_time_ms:         26.7785
- dram_time_ms:             17.8523
- compute_time_ms_lofi:     0.5402
- compute_time_ms_hifi2:    1.0805
- compute_time_ms_hifi3:    1.6207
- compute_time_ms_hifi4:    2.1609

## Files changed
- tests/benchmark/test_llms.py (added test_bartowski_glm_4_9b_chat_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
