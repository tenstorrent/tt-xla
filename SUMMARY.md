loader_path: third_party.tt_forge_models.grm_1_5b_i1_gguf.causal_lm.pytorch.loader
variant_id: Q4_K_M_GGUF
arch: p150
status: DONE_PASS
test_function: test_grm_1_5b_i1_q4_k_m_gguf
samples_per_second: 9.628359750060165
ttft_ms: 541.291846
prefill_pcc: 0.972905
first_decode_pcc: 0.990752
top_perf_samples_per_sec: 206.5544
pct_of_target: 4.7
roofline_bound: dram
optimization_level: 0
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_grm_1_5b_i1_q4_k_m_gguf

## Test
tests/benchmark/test_llms.py::test_grm_1_5b_i1_q4_k_m_gguf

## Model
- HF name:    mradermacher/GRM-1.5b-i1-GGUF
- Loader:     third_party.tt_forge_models.grm_1_5b_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GRM_1_5B_I1_Q4_K_M_GGUF

## Test config landed
- optimization_level:        0
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  9.628359750060165
- TTFT (ms):          541.291846
- Prefill PCC:        0.972905
- First decode PCC:   0.990752
- Wall clock:         0:02:40
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_grm_1_5b_i1_q4_k_m_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 4.7% (9.628 / 206.5544)

Note: optimization_level=0 was required because:
  - optimization_level=2 fails with compiler error: ttnn.paged_update_cache
    requires sharded input tensor; fallback solver exhausted 10000 combinations.
  - optimization_level=1 fails PCC (0.934 < 0.94 required).
The low % of target (4.7%) reflects optimization_level=0 leaving all tensors
in DRAM (interleaved). Higher optimization levels are blocked by the above
compiler issue.

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
- total_flops:             99494920320
- breakdown.matmul:        93855940736
- breakdown.linear:        5638979584
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        58720256
- memory_bytes: 117440512
- memory_gb:    0.109375

### Params
- count:                  1777088198
- effective_count:        1543714502
- memory_bytes:           2107080468
- memory_gb:              1.9623716063797474
- effective_memory_bytes: 1640333076
- effective_memory_gb:    1.5276792235672474
- embedding_count:        233373696
- embedding_memory_bytes: 466747392

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 206.5544
- top_perf_time_ms:         4.8413
- dram_time_ms:             3.2276
- compute_time_ms_lofi:     0.1131
- compute_time_ms_hifi2:    0.2261
- compute_time_ms_hifi3:    0.3392
- compute_time_ms_hifi4:    0.4522

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
