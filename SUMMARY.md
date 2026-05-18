loader_path: third_party.tt_forge_models.helium.causal_lm.pytorch.loader
variant_id: 1_2B
arch: p150
status: DONE_PASS
test_function: test_helium_1_2b
samples_per_second: 81.61851967015696
ttft_ms: 307.278635
prefill_pcc: 0.995877
first_decode_pcc: 0.985579
top_perf_samples_per_sec: 155.5639
pct_of_target: 52.5
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_helium_1_2b

## Test
tests/benchmark/test_llms.py::test_helium_1_2b

## Model
- HF name:    kyutai/helium-1-2b
- Loader:     third_party.tt_forge_models.helium.causal_lm.pytorch.loader
- Variant:    ModelVariant._1_2B (value: "1_2B")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  81.61851967015696
- TTFT (ms):          307.278635
- Prefill PCC:        0.995877
- First decode PCC:   0.985579
- Wall clock:         0:08:55
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_helium_1_2b_perf_metrics_3.json
Achieved vs top_perf_samples_per_sec: 52.5% (81.62 / 155.56)

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
- total_flops:             121131499648
- breakdown.matmul:        121131499648
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        234881024
- memory_bytes: 469762048
- memory_gb:    0.4375

### Params
- count:                  2023868611
- effective_count:        1892796611
- memory_bytes:           2273350408
- memory_gb:              2.1172225549817085
- effective_memory_bytes: 2011206408
- effective_memory_gb:    1.8730819299817085
- embedding_count:        131072000
- embedding_memory_bytes: 262144000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 155.5639
- top_perf_time_ms:         6.4282
- dram_time_ms:             4.2855
- compute_time_ms_lofi:     0.1376
- compute_time_ms_hifi2:    0.2753
- compute_time_ms_hifi3:    0.4129
- compute_time_ms_hifi4:    0.5506

## Files changed
- tests/benchmark/test_llms.py (added test_helium_1_2b)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed missing hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
