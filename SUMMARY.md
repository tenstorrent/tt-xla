loader_path: third_party.tt_forge_models.llama_300m_v2_bigram.causal_lm.pytorch.loader
variant_id: base
arch: p150
status: DONE_PASS
test_function: test_llama_300m_v2_bigram
samples_per_second: 91.867
ttft_ms: 77.742
prefill_pcc: 0.999435
first_decode_pcc: 0.999097
top_perf_samples_per_sec: 795.5241
pct_of_target: 11.5
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_llama_300m_v2_bigram

## Test
tests/benchmark/test_llms.py::test_llama_300m_v2_bigram

## Model
- HF name:    deqing/llama-300M-v2-bigram
- Loader:     third_party.tt_forge_models.llama_300m_v2_bigram.causal_lm.pytorch.loader
- Variant:    base

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  91.867
- TTFT (ms):          77.742
- Prefill PCC:        0.999435
- First decode PCC:   0.999097
- Wall clock:         0:01:54
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_llama_300m_v2_bigram_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 11.5% (91.867 / 795.5241)

Note: low pct_of_target (11.5%) is expected for a 300M compute-bound model — the
roofline peak assumes ideal utilization of all 110 worker cores; at this model
size, kernel dispatch overhead and memory latency dominate the measured time.

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
- total_flops:             368729654400
- breakdown.matmul:        368729654400
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        594
- memory_bytes: 2376

### KV cache
- count:        50331648
- memory_bytes: 100663296
- memory_gb:    0.09375

### Params
- count:                  451437731
- effective_count:        320103587
- memory_bytes:           602802824
- memory_gb:              0.5614038780331612
- effective_memory_bytes: 340134536
- effective_memory_gb:    0.31677497178316116
- embedding_count:        131334144
- embedding_memory_bytes: 262668288

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 795.5241
- top_perf_time_ms:         1.2570
- dram_time_ms:             0.7461
- compute_time_ms_lofi:     0.4190
- compute_time_ms_hifi2:    0.8380
- compute_time_ms_hifi3:    1.2570
- compute_time_ms_hifi4:    1.6760

## Files changed
- tests/benchmark/test_llms.py — added test_llama_300m_v2_bigram
- tests/benchmark/benchmarks/llm_benchmark.py — fixed hasattr guard for get_weight_dtype_config_path (general infrastructure fix)
- .github/workflows/perf-bench-matrix.json — added llama_300m_v2_bigram entry

## tt-forge-models submodule
93218a34fc9fc6a671e0e41101da470c80891b2a → 665edfd07fd85c6bebee444d73dba983986fd212
(Forward all args in _patched_load_gguf_checkpoint wrappers)
