loader_path: third_party.tt_forge_models.bielik_gguf.causal_lm.pytorch.loader
variant_id: 4.5B_V3.0_INSTRUCT_GGUF
arch: p150
status: DONE_PASS
test_function: test_bielik_4_5b_v3_instruct_gguf
samples_per_second: 25.982985209352886
ttft_ms: 436.954966
prefill_pcc: 0.988695
first_decode_pcc: 0.998309
top_perf_samples_per_sec: 68.7295
pct_of_target: 37.8
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_bielik_4_5b_v3_instruct_gguf

## Test
tests/benchmark/test_llms.py::test_bielik_4_5b_v3_instruct_gguf

## Model
- HF name:    speakleash/Bielik-4.5B-v3.0-Instruct-GGUF
- Loader:     third_party.tt_forge_models.bielik_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.BIELIK_4_5B_V3_0_INSTRUCT_GGUF

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  25.982985209352886
- TTFT (ms):          436.954966
- Prefill PCC:        0.988695
- First decode PCC:   0.998309
- Wall clock:         0:05:42
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_bielik_4_5b_v3_instruct_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 37.8% (25.98 / 68.73)

Note: optimization_level=2 fails with ttnn.paged_update_cache requiring sharded input tensor;
optimization_level=1 used instead, which keeps tensors in DRAM (interleaved).
Performance gap vs roofline is expected given optimization_level=1.

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
- total_flops:             300144394368
- breakdown.matmul:        300144394368
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        125829120
- memory_bytes: 251658240
- memory_gb:    0.234375

### Params
- count:                  4755540163
- effective_count:        4690004163
- memory_bytes:           5114434312
- memory_gb:              4.7631881311535835
- effective_memory_bytes: 4983362312
- effective_memory_gb:    4.6411178186535835
- embedding_count:        65536000
- embedding_memory_bytes: 131072000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 68.7295
- top_perf_time_ms:         14.5498
- dram_time_ms:             9.6999
- compute_time_ms_lofi:     0.3411
- compute_time_ms_hifi2:    0.6821
- compute_time_ms_hifi3:    1.0232
- compute_time_ms_hifi4:    1.3643

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py
- .github/workflows/perf-bench-matrix.json
- SUMMARY.md

## tt-forge-models submodule
no change
