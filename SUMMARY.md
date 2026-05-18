loader_path: third_party.tt_forge_models.chatgpt_5.causal_lm.pytorch.loader
variant_id: ChatGPT_5_0.5B
arch: p150
status: DONE_PASS
test_function: test_chatgpt_5_0_5b
samples_per_second: 142.8285394944472
ttft_ms: 127.642974
prefill_pcc: 0.945494
first_decode_pcc: 0.993126
top_perf_samples_per_sec: 545.7924
pct_of_target: 26.2
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
- Variant:    ModelVariant.CHATGPT_5_0_5B

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  142.8285394944472
- TTFT (ms):          127.642974
- Prefill PCC:        0.945494
- First decode PCC:   0.993126
- Wall clock:         0:03:56
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_chatgpt_5_0_5b_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 26.2% (142.83 / 545.79)

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
- memory_gb:              0.7424939945340157
- effective_memory_bytes: 524977544
- effective_memory_gb:    0.48892343789339066
- embedding_count:        136134656
- embedding_memory_bytes: 272269312

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 545.7924
- top_perf_time_ms:         1.8322
- dram_time_ms:             1.0457
- compute_time_ms_lofi:     0.6107
- compute_time_ms_hifi2:    1.2215
- compute_time_ms_hifi3:    1.8322
- compute_time_ms_hifi4:    2.4429

## Files changed
- tests/benchmark/test_llms.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
