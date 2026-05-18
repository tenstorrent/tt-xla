loader_path: third_party.tt_forge_models.bartowski_kurage_en_gguf.causal_lm.pytorch.loader
variant_id: kurage_en_GGUF
arch: p150
status: DONE_PASS
test_function: test_bartowski_kurage_en_gguf
samples_per_second: 36.19
ttft_ms: 290.19
prefill_pcc: 0.990610
first_decode_pcc: 0.989458
top_perf_samples_per_sec: 46.0567
pct_of_target: 78.6
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_bartowski_kurage_en_gguf

## Test
tests/benchmark/test_llms.py::test_bartowski_kurage_en_gguf

## Model
- HF name:    bartowski/kurage-en-GGUF
- Loader:     third_party.tt_forge_models.bartowski_kurage_en_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.BARTOWSKI_KURAGE_EN_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  36.19
- TTFT (ms):          290.19
- Prefill PCC:        0.990610
- First decode PCC:   0.989458
- Wall clock:         0:08:50
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_bartowski_kurage_en_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 78.6%

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
- total_flops:             452407001216
- breakdown.matmul:        422807863424
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
- count:                  7612634819
- effective_count:        7069128387
- memory_bytes:           8598274824
- memory_gb:              8.00776744633913
- effective_memory_bytes: 7511261960
- effective_memory_gb:    6.995407827198505
- embedding_count:        543506432
- embedding_memory_bytes: 1087012864

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 46.0567
- top_perf_time_ms:         21.7124
- dram_time_ms:             14.4749
- compute_time_ms_lofi:     0.5141
- compute_time_ms_hifi2:    1.0282
- compute_time_ms_hifi3:    1.5423
- compute_time_ms_hifi4:    2.0564

## Files changed
- tests/benchmark/test_llms.py
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: guard get_weight_dtype_config_path call with hasattr)

## tt-forge-models submodule
no change
