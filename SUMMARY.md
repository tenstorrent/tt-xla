loader_path: third_party.tt_forge_models.causallm_7b_gguf.causal_lm.pytorch.loader
variant_id: CausalLM_7B_Q4_K_M_GGUF
arch: p150
status: DONE_PASS
test_function: test_causallm_7b_q4_k_m_gguf
samples_per_second: 24.571882108601113
ttft_ms: 366.417317
prefill_pcc: 0.998198
first_decode_pcc: 0.997063
top_perf_samples_per_sec: 40.5012
pct_of_target: 60.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_causallm_7b_q4_k_m_gguf

## Test
tests/benchmark/test_llms.py::test_causallm_7b_q4_k_m_gguf

## Model
- HF name:    TheBloke/CausalLM-7B-GGUF
- Loader:     third_party.tt_forge_models.causallm_7b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.CAUSALLM_7B_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  24.571882108601113
- TTFT (ms):          366.417317
- Prefill PCC:        0.998198
- First decode PCC:   0.997063
- Wall clock:         0:09:14
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_causallm_7b_q4_k_m_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 60.7%

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
- total_flops:             454293454976
- breakdown.matmul:        454293454976
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        1073741824
- memory_bytes: 2147483648
- memory_gb:    2

### Params
- count:                  7720931523
- effective_count:        7098601667
- memory_bytes:           8787174152
- memory_gb:              8.183693654835224
- effective_memory_bytes: 7542514440
- effective_memory_gb:    7.024513967335224
- embedding_count:        622329856
- embedding_memory_bytes: 1244659712

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 40.5012
- top_perf_time_ms:         24.6906
- dram_time_ms:             16.4604
- compute_time_ms_lofi:     0.5162
- compute_time_ms_hifi2:    1.0325
- compute_time_ms_hifi3:    1.5487
- compute_time_ms_hifi4:    2.0650

## Files changed
- tests/benchmark/test_llms.py (added test_causallm_7b_q4_k_m_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: hasattr guard on get_weight_dtype_config_path)

## tt-forge-models submodule
no change
