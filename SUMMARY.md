loader_path: third_party.tt_forge_models.chatterbots_uncensored_8b_i1_gguf.causal_lm.pytorch.loader
variant_id: CHATTERBOTS_UNCENSORED_8B_I1_Q4_K_M_GGUF
arch: p150
status: DONE_PASS
test_function: test_chatterbots_uncensored_8b_i1_q4_k_m_gguf
samples_per_second: 33.53
ttft_ms: 306.63
prefill_pcc: 0.999152
first_decode_pcc: 0.998940
top_perf_samples_per_sec: 42.5799
pct_of_target: 78.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_chatterbots_uncensored_8b_i1_q4_k_m_gguf

## Test
tests/benchmark/test_llms.py::test_chatterbots_uncensored_8b_i1_q4_k_m_gguf

## Model
- HF name:    mradermacher/chatterbots-uncensored-8b-i1-GGUF
- Loader:     third_party.tt_forge_models.chatterbots_uncensored_8b_i1_gguf.causal_lm.pytorch.loader
- Variant:    CHATTERBOTS_UNCENSORED_8B_I1_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  33.53
- TTFT (ms):          306.63
- Prefill PCC:        0.999152
- First decode PCC:   0.998940
- Wall clock:         0:09:24
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_chatterbots_uncensored_8b_i1_q4_k_m_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 78.7% (33.53 / 42.58)

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
- total_flops:             480299188352
- breakdown.matmul:        480299188352
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
- count:                  8030294211
- effective_count:        7504941251
- memory_bytes:           9024956168
- memory_gb:              8.405145414173603
- effective_memory_bytes: 7974250248
- effective_memory_gb:    7.426599271595478
- embedding_count:        525352960
- embedding_memory_bytes: 1050705920

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 42.5799
- top_perf_time_ms:         23.4852
- dram_time_ms:             15.6568
- compute_time_ms_lofi:     0.5458
- compute_time_ms_hifi2:    1.0916
- compute_time_ms_hifi3:    1.6374
- compute_time_ms_hifi4:    2.1832

## Files changed
- tests/benchmark/test_llms.py (added test_chatterbots_uncensored_8b_i1_q4_k_m_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (added hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
