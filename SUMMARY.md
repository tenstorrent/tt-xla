loader_path: third_party.tt_forge_models.flickering_light_gguf.causal_lm.pytorch.loader
variant_id: 14B_GGUF
arch: p150
status: DONE_PASS
test_function: test_flickering_light_14b_gguf
samples_per_second: 16.628513036765508
ttft_ms: 575.406091
prefill_pcc: 0.997217
first_decode_pcc: 0.997501
top_perf_samples_per_sec: 23.3742
pct_of_target: 71.1
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: flickering_light_14b_gguf

## Test
tests/benchmark/test_llms.py::test_flickering_light_14b_gguf

## Model
- HF name:    mradermacher/FlickeringLight-14B-i1-GGUF
- Loader:     third_party.tt_forge_models.flickering_light_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.FLICKERING_LIGHT_14B_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  16.628513036765508
- TTFT (ms):          575.406091
- Prefill PCC:        0.997217
- First decode PCC:   0.997501
- Wall clock:         0:17:27
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_flickering_light_14b_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 71.1% (16.63 / 23.37)

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
- total_flops:             880468295808
- breakdown.matmul:        880468295808
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
- count:                  14428902595
- effective_count:        13757813955
- memory_bytes:           15960320776
- memory_gb:              14.864207036793232
- effective_memory_bytes: 14618143496
- effective_memory_gb:    13.614207036793232
- embedding_count:        671088640
- embedding_memory_bytes: 1342177280

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 23.3742
- top_perf_time_ms:         42.7822
- dram_time_ms:             28.5214
- compute_time_ms_lofi:     1.0005
- compute_time_ms_hifi2:    2.0011
- compute_time_ms_hifi3:    3.0016
- compute_time_ms_hifi4:    4.0021

## Files changed
- tests/benchmark/test_llms.py (added test_flickering_light_14b_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
