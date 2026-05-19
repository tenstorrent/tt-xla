loader_path: third_party.tt_forge_models.foundation_sec_1_1_8b_instruct.causal_lm.pytorch.loader
variant_id: Foundation_Sec_1_1_8B_Instruct
arch: p150
status: DONE_PASS
test_function: test_foundation_sec_1_1_8b_instruct
samples_per_second: 32.558
ttft_ms: 321.033
prefill_pcc: 0.998638
first_decode_pcc: 0.998244
top_perf_samples_per_sec: 42.5798
pct_of_target: 76.5
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: foundation_sec_1_1_8b_instruct

## Test
tests/benchmark/test_llms.py::test_foundation_sec_1_1_8b_instruct

## Model
- HF name:    fdtn-ai/Foundation-Sec-1.1-8B-Instruct
- Loader:     third_party.tt_forge_models.foundation_sec_1_1_8b_instruct.causal_lm.pytorch.loader
- Variant:    Foundation_Sec_1_1_8B_Instruct

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  32.558
- TTFT (ms):          321.033
- Prefill PCC:        0.998638
- First decode PCC:   0.998244
- Wall clock:         0:09:08
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_foundation_sec_1_1_8b_instruct_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 76.5% (32.558 / 42.5798)

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
- total_flops:             480300236928
- breakdown.matmul:        480300236928
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
- count:                  8030326979
- effective_count:        7504957635
- memory_bytes:           9025006344
- memory_gb:              8.405192144215107
- effective_memory_bytes: 7974267656
- effective_memory_gb:    7.426615484058857
- embedding_count:        525369344
- embedding_memory_bytes: 1050738688

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 42.5798
- top_perf_time_ms:         23.4853
- dram_time_ms:             15.6569
- compute_time_ms_lofi:     0.5458
- compute_time_ms_hifi2:    1.0916
- compute_time_ms_hifi3:    1.6374
- compute_time_ms_hifi4:    2.1832

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
