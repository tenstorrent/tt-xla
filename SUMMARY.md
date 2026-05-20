loader_path: third_party.tt_forge_models.gemma_2_gguf.causal_lm.pytorch.loader
variant_id: 2B_IT_GGUF
arch: p150
status: DONE_PASS
test_function: test_gemma_2_2b_it_gguf
samples_per_second: 17.776299601331342
ttft_ms: 557.991802
prefill_pcc: 0.996019
first_decode_pcc: 0.995587
top_perf_samples_per_sec: 116.8566
pct_of_target: 15.2
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: gemma_2_2b_it_gguf

## Test
tests/benchmark/test_llms.py::test_gemma_2_2b_it_gguf

## Model
- HF name:    bartowski/gemma-2-2b-it-GGUF
- Loader:     third_party.tt_forge_models.gemma_2_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GEMMA_2_2B_IT_GGUF (2B_IT_GGUF)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  17.776299601331342
- TTFT (ms):          557.991802
- Prefill PCC:        0.996019
- First decode PCC:   0.995587
- Wall clock:         0:39:05
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_gemma_2_2b_it_gguf_perf_metrics_2.json
Achieved vs top_perf_samples_per_sec: 15.2% (17.78 / 116.86)

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
- total_flops:             168174813440
- breakdown.matmul:        168174813440
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        34
- memory_bytes: 134

### KV cache
- count:        218103808
- memory_bytes: 436207616
- memory_gb:    0.40625

### Params
- count:                  3204166154
- effective_count:        2614342154
- memory_bytes:           3958097952
- memory_gb:              3.6862659752368927
- effective_memory_bytes: 2778449952
- effective_memory_gb:    2.5876331627368927
- embedding_count:        589824000
- embedding_memory_bytes: 1179648000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 116.8566
- top_perf_time_ms:         8.5575
- dram_time_ms:             5.7050
- compute_time_ms_lofi:     0.1617
- compute_time_ms_hifi2:    0.3234
- compute_time_ms_hifi3:    0.4851
- compute_time_ms_hifi4:    0.6468

## Infrastructure fix
- Modified tests/benchmark/benchmarks/llm_benchmark.py to guard `get_weight_dtype_config_path()` with `hasattr()` check (same defensive pattern as the runner's dynamic_torch_model_tester.py). This is a general fix for any loader that does not implement this method.

## Files changed
- tests/benchmark/test_llms.py (added test_gemma_2_2b_it_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json (added bartowski_gemma-2-2b-it-gguf and _accuracy entries)

## tt-forge-models submodule
no change
