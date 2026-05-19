loader_path: third_party.tt_forge_models.minerva_7b.causal_lm.pytorch.loader
variant_id: 7B_instruct_v1.0
arch: p150
status: DONE_PASS
test_function: test_minerva_7b
samples_per_second: 35.50495068882545
ttft_ms: 301.435199
prefill_pcc: 0.990814
first_decode_pcc: 0.999494
top_perf_samples_per_sec: 44.3805
pct_of_target: 80.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_minerva_7b

## Test
tests/benchmark/test_llms.py::test_minerva_7b

## Model
- HF name:    sapienzanlp/Minerva-7B-instruct-v1.0
- Loader:     third_party.tt_forge_models.minerva_7b.causal_lm.pytorch.loader
- Variant:    ModelVariant.MINERVA_7B_INSTRUCT_V1_0

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  35.50495068882545
- TTFT (ms):          301.435199
- Prefill PCC:        0.990814
- First decode PCC:   0.999494
- Wall clock:         0:08:18
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_minerva_7b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 80.0% (35.50 / 44.38)

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
- total_flops:             460115148928
- breakdown.matmul:        460115148928
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
- count:                  7399542979
- effective_count:        7189565635
- memory_bytes:           8059118344
- memory_gb:              7.505638845264912
- effective_memory_bytes: 7639163656
- effective_memory_gb:    7.114525564014912
- embedding_count:        209977344
- embedding_memory_bytes: 419954688

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 44.3805
- top_perf_time_ms:         22.5324
- dram_time_ms:             15.0216
- compute_time_ms_lofi:     0.5229
- compute_time_ms_hifi2:    1.0457
- compute_time_ms_hifi3:    1.5686
- compute_time_ms_hifi4:    2.0914

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
