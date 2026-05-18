loader_path: third_party.tt_forge_models.calliope_14b_unslop_b0_1_i1_gguf.causal_lm.pytorch.loader
variant_id: Calliope_14B_Unslop_b0_1_i1_GGUF
arch: p150
status: DONE_PASS
test_function: test_calliope_14b_unslop_b0_1_i1_gguf
samples_per_second: 19.307348066233107
ttft_ms: 438.00924
prefill_pcc: 0.992779
first_decode_pcc: 0.996652
top_perf_samples_per_sec: 25.1586
pct_of_target: 76.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_calliope_14b_unslop_b0_1_i1_gguf

## Test
tests/benchmark/test_llms.py::test_calliope_14b_unslop_b0_1_i1_gguf

## Model
- HF name:    mradermacher/Calliope-14B-Unslop-b0.1-i1-GGUF
- Loader:     third_party.tt_forge_models.calliope_14b_unslop_b0_1_i1_gguf.causal_lm.pytorch.loader
- Variant:    Calliope_14B_Unslop_b0_1_i1_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  19.31
- TTFT (ms):          438.01
- Prefill PCC:        0.992779
- First decode PCC:   0.996652
- Wall clock:         0:14:11
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_calliope_14b_unslop_b0_1_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 76.7% (19.31 / 25.16)

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
- total_flops:             820489748608
- breakdown.matmul:        820489748608
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        335544320
- memory_bytes: 671088640
- memory_gb:    0.625

### Params
- count:                  13477237955
- effective_count:        12820567235
- memory_bytes:           14935583496
- memory_gb:              13.91
- effective_memory_bytes: 13622242056
- effective_memory_gb:    12.69
- embedding_count:        656670720
- embedding_memory_bytes: 1313341440

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 25.1586
- top_perf_time_ms:         39.7478
- dram_time_ms:             26.4985
- compute_time_ms_lofi:     0.9324
- compute_time_ms_hifi2:    1.8647
- compute_time_ms_hifi3:    2.7971
- compute_time_ms_hifi4:    3.7295

## Files changed
- tests/benchmark/test_llms.py (new test function added)
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json (new entry added)

## tt-forge-models submodule
no change
