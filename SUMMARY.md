loader_path: third_party.tt_forge_models.solar_10_7b_instruct_v1_0_uncensored_gguf.causal_lm.pytorch.loader
variant_id: SOLAR_10_7B_INSTRUCT_V1_0_UNCENSORED_Q4_K_M_GGUF
arch: p150
status: DONE_PASS
test_function: test_solar_10_7b_instruct_v1_0_uncensored_gguf
samples_per_second: 24.358461598746068
ttft_ms: 461.431288
prefill_pcc: 0.999586
first_decode_pcc: 0.999515
top_perf_samples_per_sec: 30.081498331963743
pct_of_target: 81.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_solar_10_7b_instruct_v1_0_uncensored_gguf

## Test
tests/benchmark/test_llms.py::test_solar_10_7b_instruct_v1_0_uncensored_gguf

## Model
- HF name:    TheBloke/SOLAR-10.7B-Instruct-v1.0-uncensored-GGUF
- Loader:     third_party.tt_forge_models.solar_10_7b_instruct_v1_0_uncensored_gguf.causal_lm.pytorch.loader
- Variant:    SOLAR_10_7B_INSTRUCT_V1_0_UNCENSORED_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  24.358461598746068
- TTFT (ms):          461.431288
- Prefill PCC:        0.999586
- First decode PCC:   0.999515
- Wall clock:         0:13:08
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_solar_10_7b_instruct_v1_0_uncensored_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 81.0%

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
- total_flops:             678403506304
- breakdown.matmul:        678403506304
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        402653184
- memory_bytes: 805306368
- memory_gb:    0.75

### Params
- count:                  10731524291
- effective_count:        10600452291
- memory_bytes:           11525497608
- memory_gb:              10.7339561060071
- effective_memory_bytes: 11263353608
- effective_memory_gb:    10.4898154810071
- embedding_count:        131072000
- embedding_memory_bytes: 262144000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 30.081498331963743
- top_perf_time_ms:         33.2430
- dram_time_ms:             22.1620
- compute_time_ms_lofi:     0.7709
- compute_time_ms_hifi2:    1.5418
- compute_time_ms_hifi3:    2.3127
- compute_time_ms_hifi4:    3.0837

## Files changed
- tests/benchmark/test_llms.py (added test_solar_10_7b_instruct_v1_0_uncensored_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
