loader_path: third_party.tt_forge_models.bartowski_thedrummer_cydonia_24b_v2_gguf.causal_lm.pytorch.loader
variant_id: 24B_V2_GGUF
arch: p150
status: DONE_PASS
test_function: test_bartowski_thedrummer_cydonia_24b_v2_gguf
samples_per_second: 11.672818237058543
ttft_ms: 635.904806
prefill_pcc: 0.979635
first_decode_pcc: 0.998525
top_perf_samples_per_sec: 14.244175511108486
pct_of_target: 81.9
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: bartowski_thedrummer_cydonia_24b_v2_gguf

## Test
tests/benchmark/test_llms.py::test_bartowski_thedrummer_cydonia_24b_v2_gguf

## Model
- HF name:    bartowski/TheDrummer_Cydonia-24B-v2-GGUF
- Loader:     third_party.tt_forge_models.bartowski_thedrummer_cydonia_24b_v2_gguf.causal_lm.pytorch.loader
- Variant:    CYDONIA_24B_V2_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  11.672818237058543
- TTFT (ms):          635.904806
- Prefill PCC:        0.979635
- First decode PCC:   0.998525
- Wall clock:         0:20:42
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_bartowski_thedrummer_cydonia_24b_v2_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 81.9%

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
- total_flops:             1465657589888
- breakdown.matmul:        1465657589888
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        335544320
- memory_bytes: 671088640
- memory_gb:    0.625

### Params
- count:                  23572403395
- effective_count:        22901314755
- memory_bytes:           25675213576
- memory_gb:              23.911906011402607
- effective_memory_bytes: 24333036296
- effective_memory_gb:    22.661906011402607
- embedding_count:        671088640
- embedding_memory_bytes: 1342177280

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 14.244175511108486
- top_perf_time_ms:         70.204
- dram_time_ms:             46.803
- compute_time_ms_lofi:     1.666
- compute_time_ms_hifi2:    3.331
- compute_time_ms_hifi3:    4.997
- compute_time_ms_hifi4:    6.662

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
