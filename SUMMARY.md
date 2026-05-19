loader_path: third_party.tt_forge_models.llamafactory.causal_lm.pytorch.loader
variant_id: tiny_random_Llama_3
arch: p150
status: DONE_PASS
test_function: test_llamafactory_tiny_random_llama_3
samples_per_second: 576.2116
ttft_ms: 24.497835
prefill_pcc: 1.000000
first_decode_pcc: 0.999957
top_perf_samples_per_sec: 123589.3341
pct_of_target: 0.5
roofline_bound: compute
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_llamafactory_tiny_random_llama_3

## Test
tests/benchmark/test_llms.py::test_llamafactory_tiny_random_llama_3

## Model
- HF name:    llamafactory/tiny-random-Llama-3
- Loader:     third_party.tt_forge_models.llamafactory.causal_lm.pytorch.loader
- Variant:    ModelVariant.TINY_RANDOM_LLAMA_3

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  576.2116
- TTFT (ms):          24.497835
- Prefill PCC:        1.000000
- First decode PCC:   0.999957
- Wall clock:         0:08:39
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_llamafactory_tiny_random_llama_3_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 0.5% (576.21 / 123589.33)

Note: The low pct_of_target (0.5%) is expected for this tiny toy model — the
measurement harness overhead dominates compared to the model's minimal compute.
optimization_level=2 causes a "No circular buffer with id 0 exists in Program"
compiler error; optimization_level=1 is stable.

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
- total_flops:             2373451848
- breakdown.matmul:        2373451848
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        594
- memory_bytes: 2376

### KV cache
- count:        262144
- memory_bytes: 524288
- memory_gb:    0.00048828125

### Params
- count:                  4112597
- effective_count:        2060501
- memory_bytes:           6293936
- memory_gb:              0.005861684679985046
- effective_memory_bytes: 2189744
- effective_memory_gb:    0.0020393580198287964
- embedding_count:        2052096
- embedding_memory_bytes: 4104192

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 123589.3341
- top_perf_time_ms:         0.0081
- dram_time_ms:             0.0047
- compute_time_ms_lofi:     0.0027
- compute_time_ms_hifi2:    0.0054
- compute_time_ms_hifi3:    0.0081
- compute_time_ms_hifi4:    0.0108

## Files changed
- tests/benchmark/test_llms.py (added test_llamafactory_tiny_random_llama_3)
- .github/workflows/perf-bench-matrix.json (added llamafactory_tiny_random_llama_3 entry)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: gracefully handle loaders without get_weight_dtype_config_path)

## tt-forge-models submodule
no change
