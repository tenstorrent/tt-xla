loader_path: third_party.tt_forge_models.german_gpt2.causal_lm.pytorch.loader
variant_id: german_gpt2
arch: n150
status: DONE_PASS
test_function: test_german_gpt2
samples_per_second: 164.4546665879601
ttft_ms: 85.177271
prefill_pcc: 0.998698
first_decode_pcc: 0.997911
top_perf_samples_per_sec: 1650.9017
pct_of_target: 10.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_german_gpt2

## Test
tests/benchmark/test_llms.py::test_german_gpt2

## Model
- HF name:    anonymous-german-nlp/german-gpt2
- Loader:     third_party.tt_forge_models.german_gpt2.causal_lm.pytorch.loader
- Variant:    ModelVariant.GERMAN_GPT2

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  164.4546665879601
- TTFT (ms):          85.177271
- Prefill PCC:        0.998698
- First decode PCC:   0.997911
- Wall clock:         0:02:09
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_german_gpt2_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 10.0% (164.45 / 1650.90)

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
- total_flops:             167882932224
- breakdown.matmul:        53675016192
- breakdown.linear:        114207916032
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        693
- memory_bytes: 2772

### KV cache
- count:        75497472
- memory_bytes: 150994944
- memory_gb:    0.140625

### Params
- count:                  165716100
- effective_count:        124992900
- memory_bytes:           214531388
- memory_gb:              0.19979792460799217
- effective_memory_bytes: 133084988
- effective_memory_gb:    0.12394505366683006
- embedding_count:        40723200
- embedding_memory_bytes: 81446400

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 1650.9017
- top_perf_time_ms:         0.6057
- dram_time_ms:             0.4038
- compute_time_ms_lofi:     0.1908
- compute_time_ms_hifi2:    0.3816
- compute_time_ms_hifi3:    0.5723
- compute_time_ms_hifi4:    0.7631

## Files changed
- tests/benchmark/test_llms.py (added test_german_gpt2 function)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr check)
- .github/workflows/perf-bench-matrix.json (added anonymous-german-nlp_german-gpt2 entry)

## tt-forge-models submodule
no change
