loader_path: third_party.tt_forge_models.bielik_gguf.causal_lm.pytorch.loader
variant_id: 4.5B_V3.0_INSTRUCT_GGUF
arch: n150
status: DONE_PASS
test_function: test_bielik_gguf_4_5b_v3_0_instruct
samples_per_second: 21.537
ttft_ms: 966.537
prefill_pcc: 0.980783
first_decode_pcc: 0.991999
top_perf_samples_per_sec: 38.6604
pct_of_target: 55.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_bielik_gguf_4_5b_v3_0_instruct

## Test
tests/benchmark/test_llms.py::test_bielik_gguf_4_5b_v3_0_instruct

## Model
- HF name:    speakleash/Bielik-4.5B-v3.0-Instruct-GGUF
- Loader:     third_party.tt_forge_models.bielik_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.BIELIK_4_5B_V3_0_INSTRUCT_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  21.537
- TTFT (ms):          966.537
- Prefill PCC:        0.980783
- First decode PCC:   0.991999
- Wall clock:         0:27:57
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_bielik_gguf_4_5b_v3_0_instruct_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 55.7%

### System
- arch:                        wormhole_b0
- chip_count_in_system_desc:   2
- single_chip_assumption:      True
- worker_grid_cores:           64
- dram_bandwidth_bytes_per_sec: 288000000000

### Peak FLOPs
- lofi:  256000000000000
- hifi2: 128000000000000
- hifi3: 85333333333333
- hifi4: 64000000000000

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
- top_perf_samples_per_sec: 38.6604
- top_perf_time_ms:         25.8663
- dram_time_ms:             17.2442
- compute_time_ms_lofi:     1.1724
- compute_time_ms_hifi2:    2.3449
- compute_time_ms_hifi3:    3.5173
- compute_time_ms_hifi4:    4.6898

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
