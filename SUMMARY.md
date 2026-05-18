loader_path: third_party.tt_forge_models.apriel_guard_i1_gguf.causal_lm.pytorch.loader
variant_id: AprielGuard_i1_GGUF
arch: p150
status: DONE_PASS
test_function: test_apriel_guard_i1_gguf
samples_per_second: 33.37
ttft_ms: 300.7
prefill_pcc: 0.998254
first_decode_pcc: 0.994468
top_perf_samples_per_sec: 44.6332
pct_of_target: 74.8
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_apriel_guard_i1_gguf

## Test
tests/benchmark/test_llms.py::test_apriel_guard_i1_gguf

## Model
- HF name:    mradermacher/AprielGuard-i1-GGUF
- Loader:     third_party.tt_forge_models.apriel_guard_i1_gguf.causal_lm.pytorch.loader
- Variant:    AprielGuard_i1_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  33.37
- TTFT (ms):          300.7
- Prefill PCC:        0.998254
- First decode PCC:   0.994468
- Wall clock:         0:09:22
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_apriel_guard_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 74.8% (33.37 / 44.63)

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
- total_flops:             461708984448
- breakdown.matmul:        461708984448
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        201326592
- memory_bytes: 402653184
- memory_gb:    0.375

### Params
- count:                  7885542595
- effective_count:        7214453955
- memory_bytes:           9007770376
- memory_gb:              8.389139898121357
- effective_memory_bytes: 7665593096
- effective_memory_gb:    7.139139898121357
- embedding_count:        671088640
- embedding_memory_bytes: 1342177280

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 44.6332
- top_perf_time_ms:         22.4049
- dram_time_ms:             14.9366
- compute_time_ms_lofi:     0.5247
- compute_time_ms_hifi2:    1.0493
- compute_time_ms_hifi3:    1.5740
- compute_time_ms_hifi4:    2.0987

## Files changed
- tests/benchmark/test_llms.py (added test_apriel_guard_i1_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
