loader_path: third_party.tt_forge_models.qwen_1_5_gguf.causal_lm.pytorch.loader
variant_id: 14B_Chat_GGUF
arch: p150
status: DONE_PASS
test_function: test_qwen_1_5_14b_chat_gguf
samples_per_second: 14.1
ttft_ms: 581.044
prefill_pcc: 0.998479
first_decode_pcc: 0.985233
top_perf_samples_per_sec: 21.9687
pct_of_target: 64.2
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_qwen_1_5_14b_chat_gguf

## Test
tests/benchmark/test_llms.py::test_qwen_1_5_14b_chat_gguf

## Model
- HF name:    Qwen/Qwen1.5-14B-Chat-GGUF
- Loader:     third_party.tt_forge_models.qwen_1_5_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.QWEN_1_5_14B_CHAT_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  14.1
- TTFT (ms):          581.044
- Prefill PCC:        0.998479
- First decode PCC:   0.985233
- Wall clock:         0:13:44
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_qwen_1_5_14b_chat_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 64.2%

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
- total_flops:             856832082048
- breakdown.matmul:        655485829248
- breakdown.linear:        201346252800
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        1677721600
- memory_bytes: 3355443200
- memory_gb:    3.125

### Params
- count:                  14167291075
- effective_count:        13388723395
- memory_bytes:           15783619336
- memory_gb:              14.69964099675417
- effective_memory_bytes: 14226483976
- effective_memory_gb:    13.24944568425417
- embedding_count:        778567680
- embedding_memory_bytes: 1557135360

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 21.9687
- top_perf_time_ms:         45.5193
- dram_time_ms:             30.3462
- compute_time_ms_lofi:     0.9737
- compute_time_ms_hifi2:    1.9473
- compute_time_ms_hifi3:    2.9210
- compute_time_ms_hifi4:    3.8947

## Files changed
- tests/benchmark/test_llms.py (added test_qwen_1_5_14b_chat_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr check)

## tt-forge-models submodule
no change
