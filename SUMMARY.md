loader_path: third_party.tt_forge_models.bella_bartender_heretic_1b_i1_gguf.causal_lm.pytorch.loader
variant_id: 1B_i1_GGUF
arch: p150
status: DONE_PASS
test_function: test_bella_bartender_heretic_1b_i1_gguf
samples_per_second: 19.04168317349128
ttft_ms: 256.617761
prefill_pcc: 0.997922
first_decode_pcc: 0.998038
top_perf_samples_per_sec: 254.0363
pct_of_target: 7.5
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_bella_bartender_heretic_1b_i1_gguf

## Test
tests/benchmark/test_llms.py::test_bella_bartender_heretic_1b_i1_gguf

## Model
- HF name:    mradermacher/bella-bartender-heretic-1b-i1-GGUF
- Loader:     third_party.tt_forge_models.bella_bartender_heretic_1b_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.BELLA_BARTENDER_HERETIC_1B_I1_GGUF (= "1B_i1_GGUF")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  19.04168317349128
- TTFT (ms):          256.617761
- Prefill PCC:        0.997922
- First decode PCC:   0.998038
- Wall clock:         0:18:23
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_bella_bartender_heretic_1b_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 7.5% (19.04 / 254.04)

### System
- arch:                        blackhole
- chip_count_in_system_desc:   1
- single_chip_assumption:      True
- worker_grid_cores:           130
- dram_bandwidth_bytes_per_sec: 512000000000

### Peak FLOPs
- lofi:  1040000000000000
- hifi2: 520000000000000
- hifi3: 346666666666666
- hifi4: 260000000000000

### Compute
- total_flops:             79624667200
- breakdown.matmul:        79624667200
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        67108864
- memory_bytes: 134217728
- memory_gb:    0.125

### Params
- count:                  1498482854
- effective_count:        1235814566
- memory_bytes:           1838453396
- memory_gb:              1.7121931500732899
- effective_memory_bytes: 1313116820
- effective_memory_gb:    1.2229353375732899
- embedding_count:        262668288
- embedding_memory_bytes: 525336576

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 254.0363
- top_perf_time_ms:         3.9364
- dram_time_ms:             2.6243
- compute_time_ms_lofi:     0.0766
- compute_time_ms_hifi2:    0.1531
- compute_time_ms_hifi3:    0.2297
- compute_time_ms_hifi4:    0.3062

## Files changed
- tests/benchmark/test_llms.py (added test_bella_bartender_heretic_1b_i1_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
