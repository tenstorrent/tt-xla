loader_path: third_party.tt_forge_models.gams_9b_instruct.causal_lm.pytorch.loader
variant_id: GAMS_9B_INSTRUCT
arch: p150
status: DONE_PASS
test_function: test_gams_9b_instruct
samples_per_second: 19.864156646086904
ttft_ms: 606.462153
prefill_pcc: 0.999620
first_decode_pcc: 0.995405
top_perf_samples_per_sec: 33.2775
pct_of_target: 59.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_gams_9b_instruct

## Test
tests/benchmark/test_llms.py::test_gams_9b_instruct

## Model
- HF name:    cjvt/GaMS-9B-Instruct
- Loader:     third_party.tt_forge_models.gams_9b_instruct.causal_lm.pytorch.loader
- Variant:    ModelVariant.GAMS_9B_INSTRUCT

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  19.864156646086904
- TTFT (ms):          606.462153
- Prefill PCC:        0.999620
- First decode PCC:   0.995405
- Wall clock:         0:23:05
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_gams_9b_instruct_perf_metrics_2.json
Achieved vs top_perf_samples_per_sec: 59.7%

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
- total_flops:             591430418688
- breakdown.matmul:        591430418688
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        704643072
- memory_bytes: 1409286144
- memory_gb:    1.3125

### Params
- count:                  10159210246
- effective_count:        9241706246
- memory_bytes:           11656100880
- memory_gb:              10.855589881539345
- effective_memory_bytes: 9821092880
- effective_memory_gb:    9.146605506539345
- embedding_count:        917504000
- embedding_memory_bytes: 1835008000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 33.2775
- top_perf_time_ms:         30.0503
- dram_time_ms:             20.0335
- compute_time_ms_lofi:     0.6721
- compute_time_ms_hifi2:    1.3442
- compute_time_ms_hifi3:    2.0162
- compute_time_ms_hifi4:    2.6883

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py

## tt-forge-models submodule
no change
