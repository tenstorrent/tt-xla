loader_path: third_party.tt_forge_models.gpt_fr_cased_base.causal_lm.pytorch.loader
variant_id: gpt_fr_cased_base
arch: p150
status: DONE_PASS
test_function: test_gpt_fr_cased_base
samples_per_second: 50.741
ttft_ms: 254.847
prefill_pcc: 0.999521
first_decode_pcc: 0.999792
top_perf_samples_per_sec: 242.0705
pct_of_target: 21.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: gpt_fr_cased_base

## Test
tests/benchmark/test_llms.py::test_gpt_fr_cased_base

## Model
- HF name:    asi/gpt-fr-cased-base
- Loader:     third_party.tt_forge_models.gpt_fr_cased_base.causal_lm.pytorch.loader
- Variant:    ModelVariant.GPT_FR_CASED_BASE

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  50.741
- TTFT (ms):          254.847
- Prefill PCC:        0.999521
- First decode PCC:   0.999792
- Wall clock:         0:05:45
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_gpt_fr_cased_base_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 21.0%

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
- total_flops:             64936804352
- breakdown.matmul:        5734400000
- breakdown.linear:        59202404352
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        352321536
- memory_bytes: 704643072
- memory_gb:    0.65625

### Params
- count:                  1106441860
- effective_count:        1015006852
- memory_bytes:           1262616844
- memory_gb:              1.17590356990695
- effective_memory_bytes: 1079746828
- effective_memory_gb:    1.0055925957858562
- embedding_count:        91435008
- embedding_memory_bytes: 182870016

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 242.0705
- top_perf_time_ms:         4.1310
- dram_time_ms:             2.7540
- compute_time_ms_lofi:     0.0738
- compute_time_ms_hifi2:    0.1476
- compute_time_ms_hifi3:    0.2214
- compute_time_ms_hifi4:    0.2952

## Files changed
- tests/benchmark/test_llms.py (new test_gpt_fr_cased_base function)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path call with hasattr)
- .github/workflows/perf-bench-matrix.json (new entry for gpt_fr_cased_base)

## tt-forge-models submodule
no change
