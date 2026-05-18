loader_path: third_party.tt_forge_models.fanar_1_9b_instruct.causal_lm.pytorch.loader
variant_id: 9B_Instruct
arch: p150
status: DONE_PASS
test_function: test_fanar_1_9b_instruct
samples_per_second: 21.247
ttft_ms: 596.808
prefill_pcc: 1.000000
first_decode_pcc: 0.996396
top_perf_samples_per_sec: 34.8832
pct_of_target: 60.9
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: fanar_1_9b_instruct

## Test
tests/benchmark/test_llms.py::test_fanar_1_9b_instruct

## Model
- HF name:    QCRI/Fanar-1-9B-Instruct
- Loader:     third_party.tt_forge_models.fanar_1_9b_instruct.causal_lm.pytorch.loader
- Variant:    ModelVariant.FANAR_1_9B_INSTRUCT (9B_Instruct)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  21.247
- TTFT (ms):          596.808
- Prefill PCC:        1.000000
- First decode PCC:   0.996396
- Wall clock:         0:16:40
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_fanar_1_9b_instruct_perf_metrics_2.json
Achieved vs top_perf_samples_per_sec: 60.9% (21.247 / 34.8832)

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
- total_flops:             562129010944
- breakdown.matmul:        562129010944
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        34
- memory_bytes: 134

### KV cache
- count:        704643072
- memory_bytes: 1409286144
- memory_gb:    1.3125

### Params
- count:                  9243541255
- effective_count:        8783871751
- memory_bytes:           10253982740
- memory_gb:              9.549765605479479
- effective_memory_bytes: 9334643732
- effective_memory_gb:    8.693564433604479
- embedding_count:        459669504
- embedding_memory_bytes: 919339008

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 34.8832
- top_perf_time_ms:         28.6671
- dram_time_ms:             19.1114
- compute_time_ms_lofi:     0.6388
- compute_time_ms_hifi2:    1.2776
- compute_time_ms_hifi3:    1.9163
- compute_time_ms_hifi4:    2.5551

## Files changed
- tests/benchmark/test_llms.py (added test_fanar_1_9b_instruct)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
