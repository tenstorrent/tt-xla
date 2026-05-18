loader_path: third_party.tt_forge_models.granite_3_2_2b_instruct.causal_lm.pytorch.loader
variant_id: 3.2_2B_Instruct
arch: p150
status: DONE_PASS
test_function: test_granite_3_2_2b_instruct
samples_per_second: 61.0959
ttft_ms: 260.002581
prefill_pcc: 0.997552
first_decode_pcc: 0.995401
top_perf_samples_per_sec: 122.5297
pct_of_target: 49.9
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_granite_3_2_2b_instruct

## Test
tests/benchmark/test_llms.py::test_granite_3_2_2b_instruct

## Model
- HF name:    ibm-granite/granite-3.2-2b-instruct
- Loader:     third_party.tt_forge_models.granite_3_2_2b_instruct.causal_lm.pytorch.loader
- Variant:    ModelVariant.GRANITE_3_2_2B_INSTRUCT

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  61.0959
- TTFT (ms):          260.002581
- Prefill PCC:        0.997552
- First decode PCC:   0.995401
- Wall clock:         0:08:05
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_granite_3_2_2b_instruct_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 49.9%

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
- total_flops:             162135408704
- breakdown.matmul:        162135408704
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        167772160
- memory_bytes: 335544320
- memory_gb:    0.3125

### Params
- count:                  2634201254
- effective_count:        2533531814
- memory_bytes:           2893372430
- memory_gb:              2.6946630608290434
- effective_memory_bytes: 2692033550
- effective_memory_gb:    2.5071516167372465
- embedding_count:        100669440
- embedding_memory_bytes: 201338880

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 122.5297
- top_perf_time_ms:         8.1613
- dram_time_ms:             5.4409
- compute_time_ms_lofi:     0.1842
- compute_time_ms_hifi2:    0.3685
- compute_time_ms_hifi3:    0.5527
- compute_time_ms_hifi4:    0.7370

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
