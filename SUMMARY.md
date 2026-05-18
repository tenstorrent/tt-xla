loader_path: third_party.tt_forge_models.einstein_v6_1_llama3_8b.causal_lm.pytorch.loader
variant_id: Einstein_v6_1_Llama3_8B
arch: p150
status: DONE_PASS
test_function: test_einstein_v6_1_llama3_8b
samples_per_second: 33.717
ttft_ms: 312.99
prefill_pcc: 0.979870
first_decode_pcc: 0.998958
top_perf_samples_per_sec: 42.5799
pct_of_target: 79.2
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_einstein_v6_1_llama3_8b

## Test
tests/benchmark/test_llms.py::test_einstein_v6_1_llama3_8b

## Model
- HF name:    Weyaxi/Einstein-v6.1-Llama3-8B
- Loader:     third_party.tt_forge_models.einstein_v6_1_llama3_8b.causal_lm.pytorch.loader
- Variant:    EINSTEIN_V6_1_LLAMA3_8B

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  33.717
- TTFT (ms):          312.99
- Prefill PCC:        0.979870
- First decode PCC:   0.998958
- Wall clock:         0:08:43
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_einstein_v6_1_llama3_8b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 79.2% (33.717 / 42.5799)

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
- total_flops:             480299188352
- breakdown.matmul:        480299188352
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        268435456
- memory_bytes: 536870912
- memory_gb:    0.5

### Params
- count:                  8030294211
- effective_count:        7504941251
- memory_bytes:           9024956168
- memory_gb:              8.405145414173603
- effective_memory_bytes: 7974250248
- effective_memory_gb:    7.426599271595478
- embedding_count:        525352960
- embedding_memory_bytes: 1050705920

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 42.5799
- top_perf_time_ms:         23.4852
- dram_time_ms:             15.6568
- compute_time_ms_lofi:     0.5458
- compute_time_ms_hifi2:    1.0916
- compute_time_ms_hifi3:    1.6374
- compute_time_ms_hifi4:    2.1832

## Files changed
- tests/benchmark/test_llms.py (new test function test_einstein_v6_1_llama3_8b)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: guard get_weight_dtype_config_path with hasattr)
- .github/workflows/perf-bench-matrix.json (new entry: einstein_v6_1_llama3_8b)

## tt-forge-models submodule
no change
