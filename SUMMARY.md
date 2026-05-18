loader_path: third_party.tt_forge_models.gpt2_obliterated_i1_gguf.causal_lm.pytorch.loader
variant_id: gpt2_obliterated_i1_gguf
arch: p150
status: DONE_PASS
test_function: test_gpt2_obliterated_i1_gguf
samples_per_second: 128.87
ttft_ms: 73.84
prefill_pcc: 0.993775
first_decode_pcc: 0.957984
top_perf_samples_per_sec: 1662.0049
pct_of_target: 7.8
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_gpt2_obliterated_i1_gguf

## Test
tests/benchmark/test_llms.py::test_gpt2_obliterated_i1_gguf

## Model
- HF name:    mradermacher/gpt2-OBLITERATED-i1-GGUF
- Loader:     third_party.tt_forge_models.gpt2_obliterated_i1_gguf.causal_lm.pytorch.loader
- Variant:    GPT2_OBLITERATED_I1_GGUF

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=2 causes first decode PCC regression (0.938569 < 0.94 threshold) on the full model.
optimization_level=1 passes cleanly with first_decode_pcc=0.957984.

## Measured (full model, defaults)
- Sample per second:  128.87
- TTFT (ms):          73.84
- Prefill PCC:        0.993775
- First decode PCC:   0.957984
- Wall clock:         0:00:54
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_gpt2_obliterated_i1_gguf_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 7.8% (128.87 / 1662.00)

Note: Low roofline utilization (7.8%) is expected with optimization_level=1 (DRAM-only tensors).
optimization_level=2 (SRAM placement) was required for higher throughput but caused PCC failure.

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
- total_flops:             134447972352
- breakdown.matmul:        41993945088
- breakdown.linear:        92454027264
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        561
- memory_bytes: 2244

### KV cache
- count:        75497472
- memory_bytes: 150994944
- memory_gb:    0.140625

### Params
- count:                  163037316
- effective_count:        123653508
- memory_bytes:           210429500
- memory_gb:              0.19597774371504784
- effective_memory_bytes: 131661884
- effective_memory_gb:    0.12261968478560448
- embedding_count:        39383808
- embedding_memory_bytes: 78767616

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 1662.0049
- top_perf_time_ms:         0.6017
- dram_time_ms:             0.4011
- compute_time_ms_lofi:     0.1528
- compute_time_ms_hifi2:    0.3056
- compute_time_ms_hifi3:    0.4583
- compute_time_ms_hifi4:    0.6111

## Files changed
- tests/benchmark/test_llms.py (added test_gpt2_obliterated_i1_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed missing hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added gpt2_obliterated_i1_gguf entry)

## tt-forge-models submodule
no change
