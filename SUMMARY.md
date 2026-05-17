loader_path: third_party.tt_forge_models.arc1_mini_gguf.causal_lm.pytorch.loader
variant_id: Arc1_Mini_GGUF
arch: n150
status: DONE_PASS
test_function: test_arc1_mini_gguf
samples_per_second: 18.972095370586135
ttft_ms: 750.813421
prefill_pcc: 0.999800
first_decode_pcc: 0.996722
top_perf_samples_per_sec: 44.9703
pct_of_target: 42.2
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_arc1_mini_gguf

## Test
tests/benchmark/test_llms.py::test_arc1_mini_gguf

## Model
- HF name:    meissosisai/arc1-mini
- Loader:     third_party.tt_forge_models.arc1_mini_gguf.causal_lm.pytorch.loader
- Variant:    Arc1_Mini_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  18.972095370586135
- TTFT (ms):          750.813421
- Prefill PCC:        0.999800
- First decode PCC:   0.996722
- Wall clock:         0:16:09
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_arc1_mini_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 42.2%

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
- total_flops:             247774314624
- breakdown.matmul:        247774314624
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        268435456
- memory_bytes: 536870912
- memory_gb:    0.5

### Params
- count:                  4486270148
- effective_count:        3871673540
- memory_bytes:           5343034122
- memory_gb:              4.976088294759393
- effective_memory_bytes: 4113840906
- effective_memory_gb:    3.8313129041343927
- embedding_count:        614596608
- embedding_memory_bytes: 1229193216

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 44.9703
- top_perf_time_ms:         22.2369
- dram_time_ms:             14.8246
- compute_time_ms_lofi:     0.9679
- compute_time_ms_hifi2:    1.9357
- compute_time_ms_hifi3:    2.9036
- compute_time_ms_hifi4:    3.8715

## Files changed
- tests/benchmark/test_llms.py (added test_arc1_mini_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json (added arc1_mini_gguf entry)

## tt-forge-models submodule
no change
