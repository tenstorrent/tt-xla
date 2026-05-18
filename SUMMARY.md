loader_path: third_party.tt_forge_models.chatgpt_5.causal_lm.pytorch.loader
variant_id: ChatGPT_5_0.5B
arch: n150
status: DONE_PASS
test_function: test_chatgpt_5_0_5b
samples_per_second: 75.19
ttft_ms: 273.60
prefill_pcc: 0.949162
first_decode_pcc: 0.995446
top_perf_samples_per_sec: 158.776
pct_of_target: 47.4
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_chatgpt_5_0_5b

## Test
tests/benchmark/test_llms.py::test_chatgpt_5_0_5b

## Model
- HF name:    Hack337/ChatGPT-5
- Loader:     third_party.tt_forge_models.chatgpt_5.causal_lm.pytorch.loader
- Variant:    ChatGPT_5_0.5B

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  75.19
- TTFT (ms):          273.60
- Prefill PCC:        0.949162
- First decode PCC:   0.995446
- Wall clock:         0:06:44
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_chatgpt_5_0_5b_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 47.4% (75.19 / 158.78)

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
- total_flops:             537444844608
- breakdown.matmul:        510477206592
- breakdown.linear:        26967638016
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        561
- memory_bytes: 2244

### KV cache
- count:        25165824
- memory_bytes: 50331648
- memory_gb:    0.046875

### Params
- count:                  630167587
- effective_count:        494032931
- memory_bytes:           797246856
- memory_gb:              0.742
- effective_memory_bytes: 524977544
- effective_memory_gb:    0.489
- embedding_count:        136134656
- embedding_memory_bytes: 272269312

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 158.776
- top_perf_time_ms:         6.2982
- dram_time_ms:             1.8591
- compute_time_ms_lofi:     2.0994
- compute_time_ms_hifi2:    4.1988
- compute_time_ms_hifi3:    6.2982
- compute_time_ms_hifi4:    8.3976

## Files changed
- tests/benchmark/test_llms.py (added test_chatgpt_5_0_5b)
- tests/benchmark/benchmarks/llm_benchmark.py (fix: use hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
