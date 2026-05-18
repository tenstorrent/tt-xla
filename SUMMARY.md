loader_path: third_party.tt_forge_models.bartowski_nvidia_llama_3_1_8b_ultralong_4m_instruct_gguf.causal_lm.pytorch.loader
variant_id: bartowski_nvidia_Llama_3_1_8B_UltraLong_4M_Instruct_Q4_K_M_GGUF
arch: p150
status: DONE_PASS
test_function: test_bartowski_nvidia_llama_3_1_8b_ultralong_4m_instruct_q4_k_m_gguf
samples_per_second: 31.456786861114722
ttft_ms: 333.273466
prefill_pcc: 0.992848
first_decode_pcc: 0.992029
top_perf_samples_per_sec: 42.5628
pct_of_target: 73.9
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_bartowski_nvidia_llama_3_1_8b_ultralong_4m_instruct_q4_k_m_gguf

## Test
tests/benchmark/test_llms.py::test_bartowski_nvidia_llama_3_1_8b_ultralong_4m_instruct_q4_k_m_gguf

## Model
- HF name:    bartowski/nvidia_Llama-3.1-8B-UltraLong-4M-Instruct-GGUF
- Loader:     third_party.tt_forge_models.bartowski_nvidia_llama_3_1_8b_ultralong_4m_instruct_gguf.causal_lm.pytorch.loader
- Variant:    bartowski_nvidia_Llama_3_1_8B_UltraLong_4M_Instruct_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  31.456786861114722
- TTFT (ms):          333.273466
- Prefill PCC:        0.992848
- First decode PCC:   0.992029
- Wall clock:         0:09:47
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_bartowski_nvidia_llama_3_1_8b_ultralong_4m_instruct_q4_k_m_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 73.9%

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
- total_flops:             480499466368
- breakdown.matmul:        480499466368
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
- count:                  8036552899
- effective_count:        7508070595
- memory_bytes:           9034539784
- memory_gb:              8.41407085210085
- effective_memory_bytes: 7977575176
- effective_memory_gb:    7.429695852100849
- embedding_count:        528482304
- embedding_memory_bytes: 1056964608

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 42.5628
- top_perf_time_ms:         23.4947
- dram_time_ms:             15.6631
- compute_time_ms_lofi:     0.5460
- compute_time_ms_hifi2:    1.0920
- compute_time_ms_hifi3:    1.6381
- compute_time_ms_hifi4:    2.1841

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (fix: guard get_weight_dtype_config_path with hasattr for loaders without this method)

## tt-forge-models submodule
no change
