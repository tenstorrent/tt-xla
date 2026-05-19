loader_path: third_party.tt_forge_models.gemma_2_9b_chinese_chat.causal_lm.pytorch.loader
variant_id: gemma_2_9b_chinese_chat
arch: p150
status: DONE_PASS
test_function: test_gemma_2_9b_chinese_chat
samples_per_second: 21.78
ttft_ms: 599.84
prefill_pcc: 0.998891
first_decode_pcc: 0.996029
top_perf_samples_per_sec: 33.2775
pct_of_target: 65.4
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: gemma_2_9b_chinese_chat

## Test
tests/benchmark/test_llms.py::test_gemma_2_9b_chinese_chat

## Model
- HF name:    shenzhi-wang/Gemma-2-9B-Chinese-Chat
- Loader:     third_party.tt_forge_models.gemma_2_9b_chinese_chat.causal_lm.pytorch.loader
- Variant:    gemma_2_9b_chinese_chat

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  21.78
- TTFT (ms):          599.84
- Prefill PCC:        0.998891
- First decode PCC:   0.996029
- Wall clock:         0:16:47
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_gemma_2_9b_chinese_chat_perf_metrics_2.json
Achieved vs top_perf_samples_per_sec: 65.4%

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
- total_flops:             591430418688
- breakdown.matmul:        591430418688
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        34
- memory_bytes: 134

### KV cache
- count:        704643072
- memory_bytes: 1409286144
- memory_gb:    1.3125

### Params
- count:                  10159210247
- effective_count:        9241706247
- memory_bytes:           11656100884
- memory_gb:              10.855589885264635
- effective_memory_bytes: 9821092884
- effective_memory_gb:    9.146605510264635
- embedding_count:        917504000
- embedding_memory_bytes: 1835008000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 33.2775
- top_perf_time_ms:         30.0503
- dram_time_ms:             20.0335
- compute_time_ms_lofi:     0.6721
- compute_time_ms_hifi2:    1.3442
- compute_time_ms_hifi3:    2.0162
- compute_time_ms_hifi4:    2.6883

## Files changed
- tests/benchmark/test_llms.py (added test_gemma_2_9b_chinese_chat)
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
