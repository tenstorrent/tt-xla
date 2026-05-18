loader_path: third_party.tt_forge_models.acestep.causal_lm.pytorch.loader
variant_id: acestep_5hz_lm_0_6b
arch: p150
status: DONE_PASS
test_function: test_acestep_5hz_lm_0_6b
samples_per_second: 70.629
ttft_ms: 224.291
prefill_pcc: 0.946041
first_decode_pcc: 0.954633
top_perf_samples_per_sec: 368.6819
pct_of_target: 19.2
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_acestep_5hz_lm_0_6b

## Test
tests/benchmark/test_llms.py::test_acestep_5hz_lm_0_6b

## Model
- HF name:    ACE-Step/acestep-5Hz-lm-0.6B
- Loader:     third_party.tt_forge_models.acestep.causal_lm.pytorch.loader
- Variant:    ModelVariant.ACESTEP_5HZ_LM_0_6B

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  70.629
- TTFT (ms):          224.291
- Prefill PCC:        0.946041
- First decode PCC:   0.954633
- Wall clock:         0:07:43
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_acestep_5hz_lm_0_6b_perf_metrics_3.json
Achieved vs top_perf_samples_per_sec: 19.2% (70.629 / 368.6819)

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
- total_flops:             42420404352
- breakdown.matmul:        42420404352
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
- count:                  885301443
- effective_count:        662884547
- memory_bytes:           1149210632
- memory_gb:              1.0702858045697212
- effective_memory_bytes: 704376840
- effective_memory_gb:    0.656002052128315
- embedding_count:        222416896
- embedding_memory_bytes: 444833792

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 368.6819
- top_perf_time_ms:         2.7124
- dram_time_ms:             1.8082
- compute_time_ms_lofi:     0.0482
- compute_time_ms_hifi2:    0.0964
- compute_time_ms_hifi3:    0.1446
- compute_time_ms_hifi4:    0.1928

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
