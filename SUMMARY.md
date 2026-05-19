loader_path: third_party.tt_forge_models.liquid_v1_7b.causal_lm.pytorch.loader
variant_id: Liquid_V1_7B
arch: p150
status: DONE_PASS
test_function: test_liquid_v1_7b
samples_per_second: 25.525246964456763
ttft_ms: 440.741502
prefill_pcc: 0.982125
first_decode_pcc: 0.998897
top_perf_samples_per_sec: 34.832348296344755
pct_of_target: 73.3
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_liquid_v1_7b

## Test
tests/benchmark/test_llms.py::test_liquid_v1_7b

## Model
- HF name:    Junfeng5/Liquid_V1_7B
- Loader:     third_party.tt_forge_models.liquid_v1_7b.causal_lm.pytorch.loader
- Variant:    Liquid_V1_7B

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  25.525246964456763
- TTFT (ms):          440.741502
- Prefill PCC:        0.982125
- First decode PCC:   0.998897
- Wall clock:         0:11:20
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_liquid_v1_7b_perf_metrics_2.json
Achieved vs top_perf_samples_per_sec: 73.3%

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
- total_flops:             548010983680
- breakdown.matmul:        548010983680
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        939524096
- memory_bytes: 1879048192
- memory_gb:    1.75

### Params
- count:                  9374444805
- effective_count:        8562846981
- memory_bytes:           10721735694
- memory_gb:              9.985394490882754
- effective_memory_bytes: 9098540046
- effective_memory_gb:    8.473675740882754
- embedding_count:        811597824
- embedding_memory_bytes: 1623195648

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 34.832348296344755
- top_perf_time_ms:         28.7089
- dram_time_ms:             19.1393
- compute_time_ms_lofi:     0.6227
- compute_time_ms_hifi2:    1.2455
- compute_time_ms_hifi3:    1.8682
- compute_time_ms_hifi4:    2.4910

## Files changed
- tests/benchmark/test_llms.py (added test_liquid_v1_7b)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed get_weight_dtype_config_path AttributeError for loaders without this method)

## tt-forge-models submodule
no change
