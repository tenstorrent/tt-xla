loader_path: third_party.tt_forge_models.sailor2.causal_lm.pytorch.loader
variant_id: 1B_Chat
arch: p150
status: DONE_PASS
test_function: test_sailor2_1b_chat
samples_per_second: 84.35
ttft_ms: 229.3
prefill_pcc: 0.992376
first_decode_pcc: 0.994538
top_perf_samples_per_sec: 316.5097
pct_of_target: 26.7
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_sailor2_1b_chat

## Test
tests/benchmark/test_llms.py::test_sailor2_1b_chat

## Model
- HF name:    sail/Sailor2-1B-Chat
- Loader:     third_party.tt_forge_models.sailor2.causal_lm.pytorch.loader
- Variant:    ModelVariant.SAILOR2_1B_CHAT

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  84.35
- TTFT (ms):          229.3
- Prefill PCC:        0.992376
- First decode PCC:   0.994538
- Wall clock:         0:07:35
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_sailor2_1b_chat_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 26.7% (84.35 / 316.51)

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
- total_flops:             926775182400
- breakdown.matmul:        872839906368
- breakdown.linear:        53935276032
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        561
- memory_bytes: 2244

### KV cache
- count:        50331648
- memory_bytes: 100663296
- memory_gb:    0.09375

### Params
- count:                  988064803
- effective_count:        851930147
- memory_bytes:           1177578888
- memory_gb:              1.0967058017849922
- effective_memory_bytes: 905309576
- effective_memory_gb:    0.8431352451443672
- embedding_count:        136134656
- embedding_memory_bytes: 272269312

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 316.5097
- top_perf_time_ms:         3.1595
- dram_time_ms:             1.8173
- compute_time_ms_lofi:     1.0532
- compute_time_ms_hifi2:    2.1063
- compute_time_ms_hifi3:    3.1595
- compute_time_ms_hifi4:    4.2126

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (general infrastructure fix: added hasattr check before calling get_weight_dtype_config_path)

## tt-forge-models submodule
no change
