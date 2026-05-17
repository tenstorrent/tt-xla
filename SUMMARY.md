loader_path: third_party.tt_forge_models.bartowski_goppa_ai_goppa_logillama_gguf.causal_lm.pytorch.loader
variant_id: GOPPA_LOGILLAMA_Q4_K_M_GGUF
arch: n150
status: DONE_PASS
test_function: test_bartowski_goppa_logillama_gguf
samples_per_second: 53.71
ttft_ms: 392.06
prefill_pcc: 0.998518
first_decode_pcc: 0.998241
top_perf_samples_per_sec: 142.8952
pct_of_target: 37.6
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_bartowski_goppa_logillama_gguf

## Test
tests/benchmark/test_llms.py::test_bartowski_goppa_logillama_gguf

## Model
- HF name:    bartowski/goppa-ai_Goppa-LogiLlama-GGUF
- Loader:     third_party.tt_forge_models.bartowski_goppa_ai_goppa_logillama_gguf.causal_lm.pytorch.loader
- Variant:    GOPPA_LOGILLAMA_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  53.71
- TTFT (ms):          392.06
- Prefill PCC:        0.998518
- First decode PCC:   0.998241
- Wall clock:         0:09:53
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_bartowski_goppa_logillama_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 37.6% (53.71 / 142.90)

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
- total_flops:             79087927360
- breakdown.matmul:        79087927360
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
- count:                  1498486947
- effective_count:        1235816611
- memory_bytes:           1838459656
- memory_gb:              1.712198980152607
- effective_memory_bytes: 1313118984
- effective_memory_gb:    1.2229373529553413
- embedding_count:        262670336
- embedding_memory_bytes: 525340672

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 142.8952
- top_perf_time_ms:         6.9981
- dram_time_ms:             4.6654
- compute_time_ms_lofi:     0.3089
- compute_time_ms_hifi2:    0.6179
- compute_time_ms_hifi3:    0.9268
- compute_time_ms_hifi4:    1.2357

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change — submodule at 706546ab8e
