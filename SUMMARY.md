loader_path: third_party.tt_forge_models.gemma_2_9b_it_gguf.causal_lm.pytorch.loader
variant_id: 9B_IT_GGUF
arch: p150
status: DONE_PASS
test_function: test_gemma_2_9b_it_gguf
samples_per_second: 22.015435316939282
ttft_ms: 675.918187
prefill_pcc: 0.991751
first_decode_pcc: 0.993365
top_perf_samples_per_sec: 15.499146437211579
pct_of_target: 142.0
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_gemma_2_9b_it_gguf

## Test
tests/benchmark/test_llms.py::test_gemma_2_9b_it_gguf

## Model
- HF name:    lmstudio-community/gemma-2-9b-it-GGUF
- Loader:     third_party.tt_forge_models.gemma_2_9b_it_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GEMMA_2_9B_IT_GGUF (9B_IT_GGUF)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  22.015435316939282
- TTFT (ms):          675.918187
- Prefill PCC:        0.991751
- First decode PCC:   0.993365
- Wall clock:         0:19:44
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_gemma_2_9b_it_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 142.0% (22.02 / 15.50; model exceeds hifi3 ceiling, operating near hifi2 ceiling with bfp_bf8)

### System
- arch:                        N/A
- chip_count_in_system_desc:   N/A
- single_chip_assumption:      N/A
- worker_grid_cores:           N/A
- dram_bandwidth_bytes_per_sec: N/A

### Peak FLOPs
- lofi:  880000000000000
- hifi2: 440000000000000
- hifi3: 293333333333333
- hifi4: 220000000000000

### Compute
- total_flops:             18925773398016
- breakdown.matmul:        18925773398016
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        1057
- memory_bytes: 4226

### KV cache
- count:        704643072
- memory_bytes: 1409286144
- memory_gb:    1.3125

### Params
- count:                  10159210247
- effective_count:        9241706247
- memory_bytes:           11656100884
- memory_gb:              10.855589885264635
- effective_memory_bytes: 9821092884
- effective_memory_gb:    9.146605510264635
- embedding_count:        917504000
- embedding_memory_bytes: 1835008000

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 15.499146437211579
- top_perf_time_ms:         64.5196820386909
- dram_time_ms:             20.033540498046875
- compute_time_ms_lofi:     21.506560679563634
- compute_time_ms_hifi2:    43.01312135912727
- compute_time_ms_hifi3:    64.51968203869097
- compute_time_ms_hifi4:    86.02624271825454

## Files changed
- tests/benchmark/test_llms.py (added test_gemma_2_9b_it_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (fixed: guard get_weight_dtype_config_path with hasattr check for loaders that don't implement it)

## tt-forge-models submodule
no change
