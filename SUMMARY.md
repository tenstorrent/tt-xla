loader_path: third_party.tt_forge_models.jan_v1_gguf.causal_lm.pytorch.loader
variant_id: 4B_GGUF
arch: p150
status: DONE_PASS
test_function: test_jan_v1_4b_gguf
samples_per_second: 35.039152205567646
ttft_ms: 321.505092
prefill_pcc: 0.997186
first_decode_pcc: 0.997904
top_perf_samples_per_sec: 76.5390
pct_of_target: 45.8
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_jan_v1_4b_gguf

## Test
tests/benchmark/test_llms.py::test_jan_v1_4b_gguf

## Model
- HF name:    janhq/Jan-v1-4B-GGUF
- Loader:     third_party.tt_forge_models.jan_v1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.JAN_V1_4B_GGUF (4B_GGUF)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  35.039152205567646
- TTFT (ms):          321.505092
- Prefill PCC:        0.997186
- First decode PCC:   0.997904
- Wall clock:         0:12:46
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_jan_v1_4b_gguf_perf_metrics_3.json
Achieved vs top_perf_samples_per_sec: 45.8% (35.04 / 76.54 samples/sec)

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
- total_flops:             257425408128
- breakdown.matmul:        257425408128
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        301989888
- memory_bytes: 603979776
- memory_gb:    0.5625

### Params
- count:                  4411424451
- effective_count:        4022468291
- memory_bytes:           5051969288
- memory_gb:              4.705013044178486
- effective_memory_bytes: 4274056968
- effective_memory_gb:    3.980525739490986
- embedding_count:        388956160
- embedding_memory_bytes: 777912320

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 76.5390
- top_perf_time_ms:         13.0652
- dram_time_ms:             8.7102
- compute_time_ms_lofi:     0.2925
- compute_time_ms_hifi2:    0.5851
- compute_time_ms_hifi3:    0.8776
- compute_time_ms_hifi4:    1.1701

## Files changed
- tests/benchmark/test_llms.py (added test_jan_v1_4b_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed get_weight_dtype_config_path for loaders without the method)
- .github/workflows/perf-bench-matrix.json (added jan_v1_4b_gguf entry)

## tt-forge-models submodule
no change
