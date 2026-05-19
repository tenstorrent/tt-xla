loader_path: third_party.tt_forge_models.phi3_5_mini_instruct_gguf.causal_lm.pytorch.loader
variant_id: LoneStriker_Phi_3_5_Mini_Instruct_Q4_K_M
arch: p150
status: DONE_PASS
test_function: test_phi3_5_mini_instruct_lonestriker_q4_k_m
samples_per_second: 22.757
ttft_ms: 522.17
prefill_pcc: 0.996283
first_decode_pcc: 0.993525
top_perf_samples_per_sec: 73.1003
pct_of_target: 31.1
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: phi3_5_mini_instruct_lonestriker_q4_k_m

## Test
tests/benchmark/test_llms.py::test_phi3_5_mini_instruct_lonestriker_q4_k_m

## Model
- HF name:    LoneStriker/Phi-3.5-mini-instruct-GGUF
- Loader:     third_party.tt_forge_models.phi3_5_mini_instruct_gguf.causal_lm.pytorch.loader
- Variant:    LoneStriker_Phi_3_5_Mini_Instruct_Q4_K_M

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  22.757
- TTFT (ms):          522.17
- Prefill PCC:        0.996283
- First decode PCC:   0.993525
- Wall clock:         0:07:54
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_phi3_5_mini_instruct_lonestriker_q4_k_m_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 31.1%

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
- total_flops:             238232272992
- breakdown.matmul:        238232272992
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        805306368
- memory_bytes: 1610612736
- memory_gb:    1.5

### Params
- count:                  3821079731
- effective_count:        3722579123
- memory_bytes:           4152429256
- memory_gb:              3.867251105606556
- effective_memory_bytes: 3955428040
- effective_memory_gb:    3.683779425919056
- embedding_count:        98500608
- embedding_memory_bytes: 197001216

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 73.1003
- top_perf_time_ms:         13.6798
- dram_time_ms:             9.1199
- compute_time_ms_lofi:     0.2707
- compute_time_ms_hifi2:    0.5414
- compute_time_ms_hifi3:    0.8122
- compute_time_ms_hifi4:    1.0829

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: defensive hasattr check for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
