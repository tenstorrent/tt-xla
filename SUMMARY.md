loader_path: third_party.tt_forge_models.phi4_gguf.causal_lm.pytorch.loader
variant_id: mradermacher_Phi_4_Q4_K_M
arch: p150
status: DONE_PASS
test_function: test_phi4_gguf_mradermacher_phi_4_q4_k_m
samples_per_second: 15.848835195382794
ttft_ms: 599.842637
prefill_pcc: 0.998733
first_decode_pcc: 0.997841
top_perf_samples_per_sec: 22.7248
pct_of_target: 69.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_phi4_gguf_mradermacher_phi_4_q4_k_m

## Test
tests/benchmark/test_llms.py::test_phi4_gguf_mradermacher_phi_4_q4_k_m

## Model
- HF name:    mradermacher/phi-4-GGUF
- Loader:     third_party.tt_forge_models.phi4_gguf.causal_lm.pytorch.loader
- Variant:    mradermacher_Phi_4_Q4_K_M

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  15.848835195382794
- TTFT (ms):          599.842637
- Prefill PCC:        0.998733
- First decode PCC:   0.997841
- Wall clock:         ~7:00
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_phi4_gguf_mradermacher_phi_4_q4_k_m_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 69.7% (15.85 / 22.72)

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
- total_flops:             905298575488
- breakdown.matmul:        905298575488
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        419430400
- memory_bytes: 838860800
- memory_gb:    0.78125

### Params
- count:                  14659507395
- effective_count:        14145705155
- memory_bytes:           16057805576
- memory_gb:              14.954996831715107
- effective_memory_bytes: 15030201096
- effective_memory_gb:    13.997965581715107
- embedding_count:        513802240
- embedding_memory_bytes: 1027604480

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 22.7248
- top_perf_time_ms:         44.0048
- dram_time_ms:             29.3365
- compute_time_ms_lofi:     1.0287
- compute_time_ms_hifi2:    2.0575
- compute_time_ms_hifi3:    3.0862
- compute_time_ms_hifi4:    4.1150

## Files changed
- tests/benchmark/test_llms.py (added test_phi4_gguf_mradermacher_phi_4_q4_k_m)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
