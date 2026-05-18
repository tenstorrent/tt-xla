loader_path: third_party.tt_forge_models.gemma_translate_v3_12b_i1_gguf.causal_lm.pytorch.loader
variant_id: GEMMA_TRANSLATE_V3_12B_I1_Q4_K_M_GGUF
arch: p150
status: DONE_PASS
test_function: test_gemma_translate_v3_12b_i1_q4_k_m_gguf
samples_per_second: 13.521423272044757
ttft_ms: 932.040634
prefill_pcc: 0.994387
first_decode_pcc: 0.993013
top_perf_samples_per_sec: 26.328925778723903
pct_of_target: 51.4
roofline_bound: dram
optimization_level: null
trace_enabled: null
experimental_weight_dtype: null
failure_reason: null

# Benchmark added: test_gemma_translate_v3_12b_i1_q4_k_m_gguf

## Test
tests/benchmark/test_llms.py::test_gemma_translate_v3_12b_i1_q4_k_m_gguf

## Model
- HF name:    mradermacher/GemmaTranslate-v3-12B-i1-GGUF
- Loader:     third_party.tt_forge_models.gemma_translate_v3_12b_i1_gguf.causal_lm.pytorch.loader
- Variant:    GEMMA_TRANSLATE_V3_12B_I1_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2 (DEFAULT_OPTIMIZATION_LEVEL)
- trace_enabled:             true (DEFAULT_TRACE_ENABLED)
- experimental_weight_dtype: "bfp_bf8" (DEFAULT_EXPERIMENTAL_WEIGHT_DTYPE)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  13.521423272044757
- TTFT (ms):          932.040634
- Prefill PCC:        0.994387
- First decode PCC:   0.993013
- Wall clock:         0:28:38
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_gemma_translate_v3_12b_i1_q4_k_m_gguf_perf_metrics_2.json
Achieved vs top_perf_samples_per_sec: 51.4% (13.52 / 26.33)

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
- total_flops:             753137877888
- breakdown.matmul:        753137877888
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        805306368
- memory_bytes: 1610612736
- memory_gb:    1.611

### Params
- count:                  12772913157
- effective_count:        11766034437
- memory_bytes:           14517419022
- memory_gb:              14.517
- effective_memory_bytes: 12503661582
- effective_memory_gb:    12.504
- embedding_count:        1006878720
- embedding_memory_bytes: 2013757440

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 26.328925778723903
- top_perf_time_ms:         37.981
- dram_time_ms:             25.321
- compute_time_ms_lofi:     0.856
- compute_time_ms_hifi2:    1.711
- compute_time_ms_hifi3:    2.567
- compute_time_ms_hifi4:    3.423

## Files changed
- tests/benchmark/test_llms.py (new test function test_gemma_translate_v3_12b_i1_q4_k_m_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (two general fixes: layer_types override propagation to decoder layers; hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
