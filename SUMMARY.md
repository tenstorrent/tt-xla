loader_path: third_party.tt_forge_models.dream.causal_lm.pytorch.loader
variant_id: v0_Base_7B
arch: p150
status: DONE_PASS
test_function: test_dream_v0_base_7b
samples_per_second: 14.152738824083222
ttft_ms: 404.305841
prefill_pcc: 1.0
first_decode_pcc: 1.0
top_perf_samples_per_sec: 46.0472
pct_of_target: 30.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_dream_v0_base_7b

## Test
tests/benchmark/test_llms.py::test_dream_v0_base_7b

## Model
- HF name:    Dream-org/Dream-v0-Base-7B
- Loader:     third_party.tt_forge_models.dream.causal_lm.pytorch.loader
- Variant:    ModelVariant.DREAM_V0_BASE_7B

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  14.152738824083222
- TTFT (ms):          404.305841
- Prefill PCC:        1.000000
- First decode PCC:   1.000000
- Wall clock:         0:08:10
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_dream_v0_base_7b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 30.7%

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
- total_flops:             452502421632
- breakdown.matmul:        422903283840
- breakdown.linear:        29599137792
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        117440512
- memory_bytes: 234881024
- memory_gb:    0.21875

### Params
- count:                  7615616577
- effective_count:        7070619201
- memory_bytes:           8602840324
- memory_gb:              8.012019399553537
- effective_memory_bytes: 7512845572
- effective_memory_gb:    6.996882680803537
- embedding_count:        544997376
- embedding_memory_bytes: 1089994752

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 46.0472
- top_perf_time_ms:         21.7169
- dram_time_ms:             14.4779
- compute_time_ms_lofi:     0.5142
- compute_time_ms_hifi2:    1.0284
- compute_time_ms_hifi3:    1.5426
- compute_time_ms_hifi4:    2.0568

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
