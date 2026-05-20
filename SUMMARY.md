loader_path: third_party.tt_forge_models.open_calm_7b.causal_lm.pytorch.loader
variant_id: 7b
arch: p150
status: DONE_FAIL
test_function: test_open_calm_7b
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: 42.8088
pct_of_target: null
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "First decode PCC=0.872823 below required=0.94 on full 32-layer model across all optimization levels (0,1,2) and weight dtypes (bfp_bf8, none); single-layer passes PCC. Numerical accumulation issue with GPT-NeoX architecture on p150."

# Benchmark added: test_open_calm_7b

## Test
tests/benchmark/test_llms.py::test_open_calm_7b

## Model
- HF name:    cyberagent/open-calm-7b
- Loader:     third_party.tt_forge_models.open_calm_7b.causal_lm.pytorch.loader
- Variant:    ModelVariant.OPEN_CALM_7B (7b)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (DONE_FAIL)
- TTFT (ms):          N/A (DONE_FAIL)
- Prefill PCC:        0.957512 (passed on best run)
- First decode PCC:   0.872823 (failed, required 0.94) — best was opt_level=2, bfp_bf8
- Wall clock:         7:07
- Hardware:           p150

## Failure investigation
Systematic decode PCC failure across all combinations:
- opt=2 + bfp_bf8 (default):  prefill=0.9575, decode=0.8728 ← best decode PCC
- opt=1 + bfp_bf8:            prefill=0.9569, decode=0.8875
- opt=0 + bfp_bf8:            prefill=0.9638, decode=0.8186
- opt=2 + no weight dtype:    prefill=0.9728, decode=0.7811
Single-layer (num_layers=1) passes at all levels. Root cause is likely numerical accumulation across 32 GPT-NeoX layers that requires a compiler or model fix outside the scope of this skill.

## Decode roofline (first decode graph, single-chip)
Source JSON: tests/benchmark/tt_xla_open_calm_7b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: N/A (test did not pass PCC)

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
- total_flops:             426044817536
- breakdown.matmul:        13690208384
- breakdown.linear:        412354609152
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        1073741824
- memory_bytes: 2147483648
- memory_gb:    2

### Params
- count:                  6871982275
- effective_count:        6658072771
- memory_bytes:           13743964936
- memory_gb:              12.80
- effective_memory_bytes: 13316145928
- effective_memory_gb:    12.40
- embedding_count:        213909504
- embedding_memory_bytes: 427819008

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 42.8088
- top_perf_time_ms:         23.3597
- dram_time_ms:             15.5731
- compute_time_ms_lofi:     0.4841
- compute_time_ms_hifi2:    0.9683
- compute_time_ms_hifi3:    1.4524
- compute_time_ms_hifi4:    1.9366

## Files changed
- tests/benchmark/test_llms.py (added test_open_calm_7b with FAILED comment)
- tests/benchmark/benchmarks/llm_benchmark.py (general fixes: hasattr guard for get_weight_dtype_config_path; None guard for experimental_weight_dtype)

## tt-forge-models submodule
no change — submodule at 9260e41d8bee
