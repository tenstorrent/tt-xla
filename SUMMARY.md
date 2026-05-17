loader_path: third_party.tt_forge_models.bella_bartender_8b_llama3_1_i1_gguf.causal_lm.pytorch.loader
variant_id: bella_bartender_8b_llama3.1_i1_Q4_K_M_GGUF
arch: n150
status: DONE_PASS
test_function: test_bella_bartender_8b_llama3_1_i1_q4_k_m_gguf
samples_per_second: 17.55
ttft_ms: 675.42
prefill_pcc: 0.998965
first_decode_pcc: 0.998292
top_perf_samples_per_sec: 23.9513
pct_of_target: 73.3
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_bella_bartender_8b_llama3_1_i1_q4_k_m_gguf

## Test
tests/benchmark/test_llms.py::test_bella_bartender_8b_llama3_1_i1_q4_k_m_gguf

## Model
- HF name:    mradermacher/bella-bartender-8b-llama3.1-i1-GGUF
- Loader:     third_party.tt_forge_models.bella_bartender_8b_llama3_1_i1_gguf.causal_lm.pytorch.loader
- Variant:    bella_bartender_8b_llama3.1_i1_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  17.55
- TTFT (ms):          675.42
- Prefill PCC:        0.998965
- First decode PCC:   0.998292
- Wall clock:         0:17:52
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_bella_bartender_8b_llama3_1_i1_q4_k_m_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 73.3% (17.55 / 23.9513)

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
- total_flops:             480298139776
- breakdown.matmul:        480298139776
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
- count:                  8030261443
- effective_count:        7504924867
- memory_bytes:           9024905992
- memory_gb:              8.4050986841321
- effective_memory_bytes: 7974232840
- effective_memory_gb:    7.426583059132099
- embedding_count:        525336576
- embedding_memory_bytes: 1050673152

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 23.9513
- top_perf_time_ms:         41.7514
- dram_time_ms:             27.8343
- compute_time_ms_lofi:     1.8762
- compute_time_ms_hifi2:    3.7523
- compute_time_ms_hifi3:    5.6285
- compute_time_ms_hifi4:    7.5047

## Files changed
- tests/benchmark/test_llms.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
