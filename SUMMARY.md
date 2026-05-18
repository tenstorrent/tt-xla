loader_path: third_party.tt_forge_models.apertus_sea_lion_v4_8b_it.causal_lm.pytorch.loader
variant_id: default
arch: p150
status: DONE_PASS
test_function: test_apertus_sea_lion_v4_8b_it
samples_per_second: 7.7709727714594665
ttft_ms: 773.257513
prefill_pcc: 0.943017
first_decode_pcc: 0.992718
top_perf_samples_per_sec: 42.5169
pct_of_target: 18.3
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: apertus_sea_lion_v4_8b_it

## Test
tests/benchmark/test_llms.py::test_apertus_sea_lion_v4_8b_it

## Model
- HF name:    aisingapore/Apertus-SEA-LION-v4-8B-IT
- Loader:     third_party.tt_forge_models.apertus_sea_lion_v4_8b_it.causal_lm.pytorch.loader
- Variant:    ModelVariant.DEFAULT

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  7.77
- TTFT (ms):          773.26
- Prefill PCC:        0.943017
- First decode PCC:   0.992718
- Wall clock:         0:32:30
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_apertus_sea_lion_v4_8b_it_perf_metrics_2.json
Achieved vs top_perf_samples_per_sec: 18.3% (7.77 / 42.52)

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
- total_flops:             481036337280
- breakdown.matmul:        481036337280
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
- count:                  8053338436
- effective_count:        7516467524
- memory_bytes:           9060246538
- memory_gb:              8.438012132421136
- effective_memory_bytes: 7986504714
- effective_memory_gb:    7.438012132421136
- embedding_count:        536870912
- embedding_memory_bytes: 1073741824

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 42.5169
- top_perf_time_ms:         23.5201
- dram_time_ms:             15.6800
- compute_time_ms_lofi:     0.5466
- compute_time_ms_hifi2:    1.0933
- compute_time_ms_hifi3:    1.6399
- compute_time_ms_hifi4:    2.1865

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
