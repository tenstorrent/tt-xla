loader_path: third_party.tt_forge_models.nix3_i1_gguf.causal_lm.pytorch.loader
variant_id: Nix3_i1_Q4_K_M_GGUF
arch: p150
status: DONE_PASS
test_function: test_nix3_i1_q4_k_m_gguf
samples_per_second: 60.179349791700915
ttft_ms: 208.028849
prefill_pcc: 0.998506
first_decode_pcc: 0.998607
top_perf_samples_per_sec: 169.2648
pct_of_target: 35.6
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_nix3_i1_q4_k_m_gguf

## Test
tests/benchmark/test_llms.py::test_nix3_i1_q4_k_m_gguf

## Model
- HF name:    mradermacher/Nix3-i1-GGUF
- Loader:     third_party.tt_forge_models.nix3_i1_gguf.causal_lm.pytorch.loader
- Variant:    NIX3_I1_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  60.179349791700915
- TTFT (ms):          208.028849
- Prefill PCC:        0.998506
- First decode PCC:   0.998607
- Wall clock:         0:05:51
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_nix3_i1_q4_k_m_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 35.6%

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
- total_flops:             110108868736
- breakdown.matmul:        110108868736
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        234881024
- memory_bytes: 469762048
- memory_gb:    0.4375

### Params
- count:                  2031740099
- effective_count:        1720575171
- memory_bytes:           2450557704
- memory_gb:              2.2822597101330757
- effective_memory_bytes: 1828227848
- effective_memory_gb:    1.7026698663830757
- embedding_count:        311164928
- embedding_memory_bytes: 622329856

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 169.2648
- top_perf_time_ms:         5.9079
- dram_time_ms:             3.9386
- compute_time_ms_lofi:     0.1251
- compute_time_ms_hifi2:    0.2502
- compute_time_ms_hifi3:    0.3754
- compute_time_ms_hifi4:    0.5005

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: added hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
