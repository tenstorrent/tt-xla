loader_path: third_party.tt_forge_models.gallm.causal_lm.pytorch.loader
variant_id: multi_14B_v0.1
arch: p150
status: DONE_PASS
test_function: test_gallm_multi_14b_v0_1
samples_per_second: 16.357837853226425
ttft_ms: 542.398577
prefill_pcc: 0.997364
first_decode_pcc: 0.996935
top_perf_samples_per_sec: 22.9948
pct_of_target: 71.1
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_gallm_multi_14b_v0_1

## Test
tests/benchmark/test_llms.py::test_gallm_multi_14b_v0_1

## Model
- HF name:    CjangCjengh/GaLLM-multi-14B-v0.1
- Loader:     third_party.tt_forge_models.gallm.causal_lm.pytorch.loader
- Variant:    ModelVariant.GALLM_MULTI_14B_V0_1

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  16.357837853226425
- TTFT (ms):          542.398577
- Prefill PCC:        0.997364
- First decode PCC:   0.996935
- Wall clock:         0:16:41
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_gallm_multi_14b_v0_1_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 71.1%

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
- total_flops:             895411028096
- breakdown.matmul:        782657126528
- breakdown.linear:        112753901568
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
- count:                  14770033859
- effective_count:        13991466179
- memory_bytes:           16423856904
- memory_gb:              15.295908696949482
- effective_memory_bytes: 14866721544
- effective_memory_gb:    13.845713384449482
- embedding_count:        778567680
- embedding_memory_bytes: 1557135360

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 22.9948
- top_perf_time_ms:         43.4881
- dram_time_ms:             28.9921
- compute_time_ms_lofi:     1.0175
- compute_time_ms_hifi2:    2.0350
- compute_time_ms_hifi3:    3.0525
- compute_time_ms_hifi4:    4.0701

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py

## tt-forge-models submodule
no change
