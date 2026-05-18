loader_path: third_party.tt_forge_models.embodied_brain_7b_gguf.causal_lm.pytorch.loader
variant_id: 7B_GGUF
arch: p150
status: DONE_PASS
test_function: test_embodied_brain_7b_gguf
samples_per_second: 4.1723
ttft_ms: 1243.89
prefill_pcc: 0.994665
first_decode_pcc: 0.996928
top_perf_samples_per_sec: 46.0471
pct_of_target: 9.1
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_embodied_brain_7b_gguf

## Test
tests/benchmark/test_llms.py::test_embodied_brain_7b_gguf

## Model
- HF name:    ZTE-AIM/EmbodiedBrain-7B
- Loader:     third_party.tt_forge_models.embodied_brain_7b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.EMBODIED_BRAIN_7B_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  4.1723
- TTFT (ms):          1243.89
- Prefill PCC:        0.994665
- First decode PCC:   0.996928
- Wall clock:         0:06:09 (warm kernel cache; first compile: ~3h 26m total)
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_embodied_brain_7b_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 9.1% (4.17 / 46.05)

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
- total_flops:             454146600960
- breakdown.matmul:        424547463168
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
- count:                  7615622790
- effective_count:        7070625414
- memory_bytes:           8602865172
- memory_gb:              8.012042541056871
- effective_memory_bytes: 7512870420
- effective_memory_gb:    6.996905822306871
- embedding_count:        544997376
- embedding_memory_bytes: 1089994752

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 46.0471
- top_perf_time_ms:         21.7169
- dram_time_ms:             14.4779
- compute_time_ms_lofi:     0.5161
- compute_time_ms_hifi2:    1.0322
- compute_time_ms_hifi3:    1.5482
- compute_time_ms_hifi4:    2.0643

## Files changed
- tests/benchmark/test_llms.py (added test_embodied_brain_7b_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (infrastructure fix: handle missing get_weight_dtype_config_path method)
- .github/workflows/perf-bench-matrix.json (added embodied_brain_7b_gguf and embodied_brain_7b_gguf_accuracy entries)

## tt-forge-models submodule
no change
