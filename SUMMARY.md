loader_path: third_party.tt_forge_models.code_qwen_1_5.causal_lm.pytorch.loader
variant_id: 7B_Chat
arch: p150
status: DONE_PASS
test_function: test_code_qwen_1_5_7b_chat
samples_per_second: 37.28
ttft_ms: 263.537142
prefill_pcc: 0.998938
first_decode_pcc: 0.999337
top_perf_samples_per_sec: 47.2440
pct_of_target: 78.9
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: code_qwen_1_5_7b_chat

## Test
tests/benchmark/test_llms.py::test_code_qwen_1_5_7b_chat

## Model
- HF name:    Qwen/CodeQwen1.5-7B-Chat
- Loader:     third_party.tt_forge_models.code_qwen_1_5.causal_lm.pytorch.loader
- Variant:    ModelVariant.CODE_QWEN_1_5_7B_CHAT

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  37.28
- TTFT (ms):          263.537142
- Prefill PCC:        0.998938
- First decode PCC:   0.999337
- Wall clock:         0:08:33
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_code_qwen_1_5_7b_chat_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 78.9% (37.28 / 47.2440)

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
- total_flops:             439769628800
- breakdown.matmul:        396814712960
- breakdown.linear:        42954915840
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        134217728
- memory_bytes: 268435456
- memory_gb:    0.25

### Params
- count:                  7250284739
- effective_count:        6871748803
- memory_bytes:           8058708744
- memory_gb:              7.505257375538349
- effective_memory_bytes: 7301636872
- effective_memory_gb:    6.800179250538349
- embedding_count:        378535936
- embedding_memory_bytes: 757071872

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 47.2440
- top_perf_time_ms:         21.1667
- dram_time_ms:             14.1111
- compute_time_ms_lofi:     0.4997
- compute_time_ms_hifi2:    0.9995
- compute_time_ms_hifi3:    1.4992
- compute_time_ms_hifi4:    1.9990

## Files changed
- tests/benchmark/test_llms.py (added test_code_qwen_1_5_7b_chat)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed missing hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added code_qwen_1_5_7b_chat entry)

## tt-forge-models submodule
no change
