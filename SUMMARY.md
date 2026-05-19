loader_path: third_party.tt_forge_models.prithivmlmods_smollm2_135m_gguf.causal_lm.pytorch.loader
variant_id: 135M_Q4_K_M_GGUF
arch: p150
status: DONE_PASS
test_function: test_prithivmlmods_smollm2_135m_gguf
samples_per_second: 164.75276253126174
ttft_ms: 127.913661
prefill_pcc: 0.994322
first_decode_pcc: 0.995915
top_perf_samples_per_sec: 1821.6176
pct_of_target: 9.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: prithivmlmods_smollm2_135m_gguf

## Test
tests/benchmark/test_llms.py::test_prithivmlmods_smollm2_135m_gguf

## Model
- HF name:    prithivMLmods/SmolLM2-135M-GGUF
- Loader:     third_party.tt_forge_models.prithivmlmods_smollm2_135m_gguf.causal_lm.pytorch.loader
- Variant:    SMOLLM2_135M_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  164.75276253126174
- TTFT (ms):          127.913661
- Prefill PCC:        0.994322
- First decode PCC:   0.995915
- Wall clock:         0:04:16
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_prithivmlmods_smollm2_135m_gguf_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 9.0%

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
- total_flops:             146314101824
- breakdown.matmul:        146314101824
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        561
- memory_bytes: 2244

### KV cache
- count:        47185920
- memory_bytes: 94371840
- memory_gb:    0.087890625

### Params
- count:                  162826723
- effective_count:        134515171
- memory_bytes:           199578888
- memory_gb:              0.18587232381105423
- effective_memory_bytes: 142955784
- effective_memory_gb:    0.13313794881105423
- embedding_count:        28311552
- embedding_memory_bytes: 56623104

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 1821.6176
- top_perf_time_ms:         0.5490
- dram_time_ms:             0.3660
- compute_time_ms_lofi:     0.1663
- compute_time_ms_hifi2:    0.3325
- compute_time_ms_hifi3:    0.4988
- compute_time_ms_hifi4:    0.6651

## Files changed
- tests/benchmark/test_llms.py
- .github/workflows/perf-bench-matrix.json

## tt-forge-models submodule
no change
