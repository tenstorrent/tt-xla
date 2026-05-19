loader_path: third_party.tt_forge_models.navila_llama3_8b_8f.causal_lm.pytorch.loader
variant_id: Llama3_8B_8F
arch: p150
status: DONE_PASS
test_function: test_navila_llama3_8b_8f
samples_per_second: 33.65
ttft_ms: 307.55
prefill_pcc: 0.999442
first_decode_pcc: 0.999060
top_perf_samples_per_sec: 42.58
pct_of_target: 79.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: test_navila_llama3_8b_8f

## Test
tests/benchmark/test_llms.py::test_navila_llama3_8b_8f

## Model
- HF name:    a8cheng/navila-llama3-8b-8f
- Loader:     third_party.tt_forge_models.navila_llama3_8b_8f.causal_lm.pytorch.loader
- Variant:    ModelVariant.NAVILA_LLAMA3_8B_8F

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  33.65
- TTFT (ms):          307.55
- Prefill PCC:        0.999442
- First decode PCC:   0.999060
- Wall clock:         0:08:19
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_navila_llama3_8b_8f_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 79.0%

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
- total_flops:             480298926208
- breakdown.matmul:        480298926208
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
- count:                  8030286019
- effective_count:        7504937155
- memory_bytes:           9024943624
- memory_gb:              8.405133731663227
- effective_memory_bytes: 7974245896
- effective_memory_gb:    7.426595218479633
- embedding_count:        525348864
- embedding_memory_bytes: 1050697728

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 42.58
- top_perf_time_ms:         23.4852
- dram_time_ms:             15.6568
- compute_time_ms_lofi:     0.5458
- compute_time_ms_hifi2:    1.0916
- compute_time_ms_hifi3:    1.6374
- compute_time_ms_hifi4:    2.1832

## Files changed
- tests/benchmark/test_llms.py (added test_navila_llama3_8b_8f)
- tests/benchmark/benchmarks/llm_benchmark.py (fix: guard get_weight_dtype_config_path call with hasattr)

## tt-forge-models submodule
no change
