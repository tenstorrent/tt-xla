loader_path: third_party.tt_forge_models.blossom_v6_7b_i1_gguf.causal_lm.pytorch.loader
variant_id: V6_7B_i1_GGUF
arch: p150
status: DONE_PASS
test_function: test_blossom_v6_7b_i1_gguf
samples_per_second: 4.196814271029099
ttft_ms: 743.201629
prefill_pcc: 0.981807
first_decode_pcc: 0.998500
top_perf_samples_per_sec: 46.0472
pct_of_target: 9.1
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: blossom_v6_7b_i1_gguf

## Test
tests/benchmark/test_llms.py::test_blossom_v6_7b_i1_gguf

## Model
- HF name:    mradermacher/Blossom-V6-7B-i1-GGUF
- Loader:     third_party.tt_forge_models.blossom_v6_7b_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.BLOSSOM_V6_7B_I1_GGUF

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=2 causes the decode graph compilation to hang indefinitely
on p150 (Blackhole). optimization_level=1 (DRAM-only, no SRAM tensor placement) avoids
the hang and achieves stable decode throughput.

## Measured (full model, defaults)
- Sample per second:  4.196814271029099
- TTFT (ms):          743.201629
- Prefill PCC:        0.981807
- First decode PCC:   0.998500
- Wall clock:         0:24:17
- Hardware:           p150 (Blackhole)

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_blossom_v6_7b_i1_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 9.1% (4.197 / 46.0472)

### System
- arch:                        blackhole
- chip_count_in_system_desc:   1
- single_chip_assumption:      True
- worker_grid_cores:           130
- dram_bandwidth_bytes_per_sec: 512000000000

### Peak FLOPs
- lofi:  1040000000000000
- hifi2: 520000000000000
- hifi3: 346666666666666
- hifi4: 260000000000000

### Compute
- total_flops:             454146588800
- breakdown.matmul:        424547451008
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
- count:                  7615616710
- effective_count:        7070619334
- memory_bytes:           8602840852
- memory_gb:              8.012019891291857
- effective_memory_bytes: 7512846100
- effective_memory_gb:    6.996883172541857
- embedding_count:        544997376
- embedding_memory_bytes: 1089994752

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 46.0472
- top_perf_time_ms:         21.7169
- dram_time_ms:             14.4779
- compute_time_ms_lofi:     0.8140
- compute_time_ms_hifi2:    0.8734
- compute_time_ms_hifi3:    1.3100
- compute_time_ms_hifi4:    1.7467

## Files changed
- tests/benchmark/test_llms.py (added test_blossom_v6_7b_i1_gguf)
- .github/workflows/perf-bench-matrix.json (added blossom_v6_7b_i1_gguf CI entry)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed get_weight_dtype_config_path to use getattr with default None — general infrastructure fix for loaders without this method)

## tt-forge-models submodule
no change
