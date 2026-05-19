loader_path: third_party.tt_forge_models.internlm2_math_plus_7b_gguf.causal_lm.pytorch.loader
variant_id: 7B_Math_Plus_Q4_K_M
arch: p150
status: DONE_PASS
test_function: test_internlm2_math_plus_7b_gguf
samples_per_second: 32.925
ttft_ms: 308.237
prefill_pcc: 0.996145
first_decode_pcc: 0.982643
top_perf_samples_per_sec: 43.3967
pct_of_target: 75.9
roofline_bound: "dram"
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: internlm2_math_plus_7b_gguf

## Test
tests/benchmark/test_llms.py::test_internlm2_math_plus_7b_gguf

## Model
- HF name:    RichardErkhov/internlm_-_internlm2-math-plus-7b-gguf
- Loader:     third_party.tt_forge_models.internlm2_math_plus_7b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.INTERNLM2_MATH_PLUS_7B_Q4_K_M

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  32.925
- TTFT (ms):          308.237
- Prefill PCC:        0.996145
- First decode PCC:   0.982643
- Wall clock:         0:10:49
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tests/benchmark/tt_xla_internlm2_math_plus_7b_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 75.9%

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
- total_flops:             470936453248
- breakdown.matmul:        470936453248
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
- count:                  7737708739
- effective_count:        7358648515
- memory_bytes:           8576934664
- memory_gb:              7.987892873585224
- effective_memory_bytes: 7818814216
- effective_memory_gb:    7.281838186085224
- embedding_count:        379060224
- embedding_memory_bytes: 758120448

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 43.3967
- top_perf_time_ms:         23.0433
- dram_time_ms:             15.3622
- compute_time_ms_lofi:     0.5352
- compute_time_ms_hifi2:    1.0703
- compute_time_ms_hifi3:    1.6055
- compute_time_ms_hifi4:    2.1406

## Files changed
- tests/benchmark/test_llms.py (added test_internlm2_math_plus_7b_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (general harness fix: hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added internlm2_math_plus_7b_gguf entry)

## tt-forge-models submodule
no change
