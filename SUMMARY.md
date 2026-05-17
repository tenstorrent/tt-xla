loader_path: third_party.tt_forge_models.bartowski_ibm_granite_granite_4_0_micro_gguf.causal_lm.pytorch.loader
variant_id: IBM_Granite_Granite_4_0_Micro_Q4_K_M_GGUF
arch: n150
status: DONE_PASS
test_function: test_ibm_granite_4_0_micro_q4_k_m_gguf
samples_per_second: 27.464
ttft_ms: 574.562
prefill_pcc: 0.997997
first_decode_pcc: 0.998598
top_perf_samples_per_sec: 52.1429
pct_of_target: 52.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_ibm_granite_4_0_micro_q4_k_m_gguf

## Test
tests/benchmark/test_llms.py::test_ibm_granite_4_0_micro_q4_k_m_gguf

## Model
- HF name:    bartowski/ibm-granite_granite-4.0-micro-GGUF
- Loader:     third_party.tt_forge_models.bartowski_ibm_granite_granite_4_0_micro_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.IBM_GRANITE_GRANITE_4_0_MICRO_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  27.464
- TTFT (ms):          574.562
- Prefill PCC:        0.997997
- First decode PCC:   0.998598
- Wall clock:         0:16:05
- Hardware:           n300 (single-chip n150 assumption)

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_ibm_granite_4_0_micro_q4_k_m_gguf_perf_metrics_3.json
Achieved vs top_perf_samples_per_sec: 52.7% (27.464 / 52.1429)

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
- total_flops:             217768263744
- breakdown.matmul:        217768263744
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        167772160
- memory_bytes: 335544320
- memory_gb:    0.3125

### Params
- count:                  3659737766
- effective_count:        3402836646
- memory_bytes:           4129511054
- memory_gb:              3.846
- effective_memory_bytes: 3615708814
- effective_memory_gb:    3.367
- embedding_count:        256901120
- embedding_memory_bytes: 513802240

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 52.1429
- top_perf_time_ms:         19.1781
- dram_time_ms:             12.7854
- compute_time_ms_lofi:     0.8507
- compute_time_ms_hifi2:    1.7013
- compute_time_ms_hifi3:    2.5520
- compute_time_ms_hifi4:    3.4026

## Files changed
- tests/benchmark/test_llms.py (added test_ibm_granite_4_0_micro_q4_k_m_gguf)
- .github/workflows/perf-bench-matrix.json (added ibm_granite_4_0_micro_q4_k_m_gguf entry)
- tests/benchmark/benchmarks/llm_benchmark.py (make get_weight_dtype_config_path optional via hasattr)

## tt-forge-models submodule
no change
