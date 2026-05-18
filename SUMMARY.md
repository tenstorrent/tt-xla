loader_path: third_party.tt_forge_models.bartowski_ministral_8b_instruct_2410_gguf.causal_lm.pytorch.loader
variant_id: bartowski_Ministral_8B_Instruct_2410_Q4_K_M_GGUF
arch: p150
status: DONE_PASS
test_function: test_bartowski_ministral_8b_instruct_2410_q4_k_m_gguf
samples_per_second: 32.383
ttft_ms: 338.968
prefill_pcc: 0.998846
first_decode_pcc: 0.997914
top_perf_samples_per_sec: 42.5168
pct_of_target: 76.2
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_bartowski_ministral_8b_instruct_2410_q4_k_m_gguf

## Test
tests/benchmark/test_llms.py::test_bartowski_ministral_8b_instruct_2410_q4_k_m_gguf

## Model
- HF name:    bartowski/Ministral-8B-Instruct-2410-GGUF
- Loader:     third_party.tt_forge_models.bartowski_ministral_8b_instruct_2410_gguf.causal_lm.pytorch.loader
- Variant:    bartowski_Ministral_8B_Instruct_2410_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  32.383
- TTFT (ms):          338.968
- Prefill PCC:        0.998846
- First decode PCC:   0.997914
- Wall clock:         0:10:27
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_bartowski_ministral_8b_instruct_2410_q4_k_m_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 76.2%

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
- total_flops:             478888853632
- breakdown.matmul:        478888853632
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        301989888
- memory_bytes: 603979776
- memory_gb:    0.5625

### Params
- count:                  8019808451
- effective_count:        7482937539
- memory_bytes:           9024643848
- memory_gb:              8.4048545435071
- effective_memory_bytes: 7950902024
- effective_memory_gb:    7.404854543507099
- embedding_count:        536870912
- embedding_memory_bytes: 1073741824

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 42.5168
- top_perf_time_ms:         23.5201
- dram_time_ms:             15.6801
- compute_time_ms_lofi:     0.5442
- compute_time_ms_hifi2:    1.0884
- compute_time_ms_hifi3:    1.6326
- compute_time_ms_hifi4:    2.1768

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
