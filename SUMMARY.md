loader_path: third_party.tt_forge_models.glm.causal_lm.pytorch.loader
variant_id: Z1_9B_0414
arch: p150
status: DONE_PASS
test_function: test_glm_z1_9b_0414
samples_per_second: 13.969066189186172
ttft_ms: 1103.557353
prefill_pcc: 0.995137
first_decode_pcc: 0.996742
top_perf_samples_per_sec: 37.3434
pct_of_target: 37.4
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_glm_z1_9b_0414

## Test
tests/benchmark/test_llms.py::test_glm_z1_9b_0414

## Model
- HF name:    zai-org/GLM-Z1-9B-0414
- Loader:     third_party.tt_forge_models.glm.causal_lm.pytorch.loader
- Variant:    ModelVariant.GLM_Z1_9B_0414

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  13.969066189186172
- TTFT (ms):          1103.557353
- Prefill PCC:        0.995137
- First decode PCC:   0.996742
- Wall clock:         0:11:52
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_glm_z1_9b_0414_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 37.4% (13.97 / 37.34)

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
- total_flops:             561841307712
- breakdown.matmul:        513517027392
- breakdown.linear:        48324280320
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        83886080
- memory_bytes: 167772160
- memory_gb:    0.15625

### Params
- count:                  9400279203
- effective_count:        8779522211
- memory_bytes:           10570547848
- memory_gb:              9.844589836895466
- effective_memory_bytes: 9329033864
- effective_memory_gb:    8.688339836895466
- embedding_count:        620756992
- embedding_memory_bytes: 1241513984

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 37.3434
- top_perf_time_ms:         26.7785
- dram_time_ms:             17.8523
- compute_time_ms_lofi:     0.6385
- compute_time_ms_hifi2:    1.2769
- compute_time_ms_hifi3:    1.9154
- compute_time_ms_hifi4:    2.5538

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
