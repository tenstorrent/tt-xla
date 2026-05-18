loader_path: third_party.tt_forge_models.bielik.causal_lm.pytorch.loader
variant_id: 11B_v2.3_Instruct
arch: p150
status: DONE_PASS
test_function: test_bielik_11b_v2_3_instruct
samples_per_second: 23.077363540245514
ttft_ms: 483.516345
prefill_pcc: 0.986266
first_decode_pcc: 0.990846
top_perf_samples_per_sec: 28.8907
pct_of_target: 79.9
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: bielik_11b_v2_3_instruct

## Test
tests/benchmark/test_llms.py::test_bielik_11b_v2_3_instruct

## Model
- HF name:    speakleash/Bielik-11B-v2.3-Instruct
- Loader:     third_party.tt_forge_models.bielik.causal_lm.pytorch.loader
- Variant:    ModelVariant.BIELIK_11B_V2_3_INSTRUCT

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  23.077363540245514
- TTFT (ms):          483.516345
- Prefill PCC:        0.986266
- First decode PCC:   0.990846
- Wall clock:         0:13:37
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_bielik_11b_v2_3_instruct_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 79.9% (23.08 / 28.89)

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
- total_flops:             706354348160
- breakdown.matmul:        706354348160
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        419430400
- memory_bytes: 838860800
- memory_gb:    0.78125

### Params
- count:                  11168796867
- effective_count:        11037200579
- memory_bytes:           11990606600
- memory_gb:              11.16712260991335
- effective_memory_bytes: 11727414024
- effective_memory_gb:    10.92200542241335
- embedding_count:        131596288
- embedding_memory_bytes: 263192576

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 28.8907
- top_perf_time_ms:         34.6132
- dram_time_ms:             23.0755
- compute_time_ms_lofi:     0.8027
- compute_time_ms_hifi2:    1.6054
- compute_time_ms_hifi3:    2.4080
- compute_time_ms_hifi4:    3.2107

## Files changed
- tests/benchmark/test_llms.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
