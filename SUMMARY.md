loader_path: third_party.tt_forge_models.kogpt.causal_lm.pytorch.loader
variant_id: Default
arch: p150
status: DONE_PASS
test_function: test_kogpt
samples_per_second: 111.61
ttft_ms: 119.76
prefill_pcc: 0.998109
first_decode_pcc: 0.997871
top_perf_samples_per_sec: 490.9606
pct_of_target: 22.7
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark added: kogpt

## Test
tests/benchmark/test_llms.py::test_kogpt

## Model
- HF name:    psyche/kogpt
- Loader:     third_party.tt_forge_models.kogpt.causal_lm.pytorch.loader
- Variant:    ModelVariant.BASE (Default)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  111.61
- TTFT (ms):          119.76
- Prefill PCC:        0.998109
- First decode PCC:   0.997871
- Wall clock:         0:02:33
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_kogpt_perf_metrics_0.json
Achieved vs top_perf_samples_per_sec: 22.7% (111.61 / 490.96)

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
- total_flops:             597468119040
- breakdown.matmul:        75502190592
- breakdown.linear:        521965928448
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        792
- memory_bytes: 3168

### KV cache
- count:        150994944
- memory_bytes: 301989888
- memory_gb:    0.28125

### Params
- count:                  441437316
- effective_count:        389136516
- memory_bytes:           518618828
- memory_gb:              0.483001422137022
- effective_memory_bytes: 414017228
- effective_memory_gb:    0.3855835907161236
- embedding_count:        52300800
- embedding_memory_bytes: 104601600

### Roofline
- bound:                    compute
- top_perf_samples_per_sec: 490.9606
- top_perf_time_ms:         2.0368
- dram_time_ms:             1.0879
- compute_time_ms_lofi:     0.6789
- compute_time_ms_hifi2:    1.3579
- compute_time_ms_hifi3:    2.0368
- compute_time_ms_hifi4:    2.7158

## Files changed
- tests/benchmark/test_llms.py (added test_kogpt)
- tests/benchmark/benchmarks/llm_benchmark.py (two general fixes: lazy tokenizer init, hasattr guard for get_weight_dtype_config_path, float32→bfloat16 model cast)
- .github/workflows/perf-bench-matrix.json (added psyche_kogpt entry)

## tt-forge-models submodule
no change
