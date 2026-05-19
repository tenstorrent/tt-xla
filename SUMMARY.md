loader_path: third_party.tt_forge_models.llama_3_8b_instruct_262k.causal_lm.pytorch.loader
variant_id: gradientai_llama_3_8b_instruct_262k
arch: p150
status: DONE_PASS
test_function: test_llama_3_8b_instruct_262k
samples_per_second: 33.69
ttft_ms: 311.74
prefill_pcc: 0.998088
first_decode_pcc: 0.998802
top_perf_samples_per_sec: 440.2389
pct_of_target: 7.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_llama_3_8b_instruct_262k

## Test
tests/benchmark/test_llms.py::test_llama_3_8b_instruct_262k

## Model
- HF name:    gradientai/Llama-3-8B-Instruct-262k
- Loader:     third_party.tt_forge_models.llama_3_8b_instruct_262k.causal_lm.pytorch.loader
- Variant:    ModelVariant.GRADIENTAI_LLAMA_3_8B_INSTRUCT_262K

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  33.69
- TTFT (ms):          311.74
- Prefill PCC:        0.998088
- First decode PCC:   0.998802
- Wall clock:         0:18:29
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_llama_3_8b_instruct_262k_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 7.7% (33.69 / 440.24)

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
- total_flops:             47580184704
- breakdown.matmul:        47580184704
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        8388608
- memory_bytes: 16777216
- memory_gb:    0.015625

### Params
- count:                  1268789443
- effective_count:        743452867
- memory_bytes:           1840603912
- memory_gb:              1.7141959741711617
- effective_memory_bytes: 789930760
- effective_memory_gb:    0.7356803491711617
- embedding_count:        525336576
- embedding_memory_bytes: 1050673152

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 440.2389
- top_perf_time_ms:         2.2715
- dram_time_ms:             1.5143
- compute_time_ms_lofi:     0.0541
- compute_time_ms_hifi2:    0.1081
- compute_time_ms_hifi3:    0.1622
- compute_time_ms_hifi4:    0.2163

## Files changed
- tests/benchmark/test_llms.py (pre-existing, no changes needed)
- .github/workflows/perf-bench-matrix.json (pre-existing, no changes needed)

## tt-forge-models submodule
no change
