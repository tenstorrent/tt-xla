loader_path: third_party.tt_forge_models.arc1_mini_gguf.causal_lm.pytorch.loader
variant_id: Arc1_Mini_GGUF
arch: p150
status: DONE_PASS
test_function: test_arc1_mini_gguf
samples_per_second: 36.72802698156255
ttft_ms: 319.675253
prefill_pcc: 0.999702
first_decode_pcc: 0.994156
top_perf_samples_per_sec: 79.9471
pct_of_target: 45.9
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
- Variant:    ModelVariant.ARC1_MINI_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  36.72802698156255
- TTFT (ms):          319.675253
- Prefill PCC:        0.999702
- First decode PCC:   0.994156
- Wall clock:         0:09:32
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_arc1_mini_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 45.9% (36.7 / 79.9 samples/sec)

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
- top_perf_samples_per_sec: 79.9471
- top_perf_time_ms:         12.5083
- dram_time_ms:             8.3388
- compute_time_ms_lofi:     0.2816
- compute_time_ms_hifi2:    0.5631
- compute_time_ms_hifi3:    0.8447
- compute_time_ms_hifi4:    1.1262

## Files changed
- tests/benchmark/test_llms.py (added test_arc1_mini_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed get_weight_dtype_config_path hasattr guard)
- .github/workflows/perf-bench-matrix.json (added arc1_mini_gguf entry)

## tt-forge-models submodule
no change
