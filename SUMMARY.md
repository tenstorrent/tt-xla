loader_path: third_party.tt_forge_models.apriel.causal_lm.pytorch.loader
variant_id: 5B_Instruct
arch: p150
status: DONE_PASS
test_function: test_apriel_5b_instruct
samples_per_second: 14.59
ttft_ms: 408.4
prefill_pcc: 0.998806
first_decode_pcc: 0.999408
top_perf_samples_per_sec: 73.0649
pct_of_target: 20.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: null
failure_reason: null

# Benchmark added: test_apriel_5b_instruct

## Test
tests/benchmark/test_llms.py::test_apriel_5b_instruct

## Model
- HF name:    ServiceNow-AI/Apriel-5B-Instruct
- Loader:     third_party.tt_forge_models.apriel.causal_lm.pytorch.loader
- Variant:    ModelVariant.APRIEL_5B_INSTRUCT

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: none (bfp_bf8 caused PCC=0.321 on the full model; reverted per policy)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  14.59
- TTFT (ms):          408.4
- Prefill PCC:        0.998806
- First decode PCC:   0.999408
- Wall clock:         0:11:39
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_apriel_5b_instruct_perf_metrics_3.json
Achieved vs top_perf_samples_per_sec: 20.0% (14.59 / 73.0649)

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
- total_flops:             274877907072
- breakdown.matmul:        274877907072
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        234881024
- memory_bytes: 469762048
- memory_gb:    0.4375

### Params
- count:                  4832071875
- effective_count:        4295200963
- memory_bytes:           9664144138
- memory_gb:              9.000435600057244
- effective_memory_bytes: 8590402314
- effective_memory_gb:    8.000435600057244
- embedding_count:        536870912
- embedding_memory_bytes: 1073741824

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 73.0649
- top_perf_time_ms:         13.6865
- dram_time_ms:             9.1243
- compute_time_ms_lofi:     0.3124
- compute_time_ms_hifi2:    0.6247
- compute_time_ms_hifi3:    0.9371
- compute_time_ms_hifi4:    1.2494

## Files changed
- tests/benchmark/test_llms.py (added test_apriel_5b_instruct)
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: hasattr guard for get_weight_dtype_config_path; None-safe experimental_weight_dtype handling)

## tt-forge-models submodule
no change
