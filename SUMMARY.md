loader_path: third_party.tt_forge_models.athena_1_7b_i1_gguf.causal_lm.pytorch.loader
variant_id: 7B_I1_GGUF
arch: p150
status: DONE_PASS
test_function: test_athena_1_7b_i1_gguf
samples_per_second: 24.164141371105277
ttft_ms: 299.164285
prefill_pcc: 0.979758
first_decode_pcc: 0.966543
top_perf_samples_per_sec: 46.0472
pct_of_target: 52.5
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: ""
failure_reason: null

# Benchmark added: athena_1_7b_i1_gguf

## Test
tests/benchmark/test_llms.py::test_athena_1_7b_i1_gguf

## Model
- HF name:    mradermacher/Athena-1-7B-i1-GGUF
- Loader:     third_party.tt_forge_models.athena_1_7b_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.ATHENA_1_7B_I1_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "" (disabled; bfp_bf8 default caused first decode PCC=0.875, below 0.94 threshold)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  24.164141371105277
- TTFT (ms):          299.164285
- Prefill PCC:        0.979758
- First decode PCC:   0.966543
- Wall clock:         0:08:28
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_athena_1_7b_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 52.5% (24.16 / 46.05)

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
- total_flops:             452502421632
- breakdown.matmul:        422903283840
- breakdown.linear:        29599137792
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        117440512
- memory_bytes: 234881024
- memory_gb:    0.21875

### Params
- count:                  7615616707
- effective_count:        7070619331
- memory_bytes:           15231233800
- memory_gb:              14.185191877186298
- effective_memory_bytes: 14141239048
- effective_memory_gb:    13.170055158436298
- embedding_count:        544997376
- embedding_memory_bytes: 1089994752

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 46.0472
- top_perf_time_ms:         21.7169
- dram_time_ms:             14.4779
- compute_time_ms_lofi:     0.5142
- compute_time_ms_hifi2:    1.0284
- compute_time_ms_hifi3:    1.5426
- compute_time_ms_hifi4:    2.0568

## Files changed
- tests/benchmark/test_llms.py

## tt-forge-models submodule
no change
