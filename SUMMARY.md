loader_path: third_party.tt_forge_models.olmo2.causal_lm.pytorch.loader
variant_id: 1124_13b
arch: p150
status: DONE_PASS
test_function: test_olmo2_1124_13b
samples_per_second: 2.2175662374426244
ttft_ms: 2481.934794
prefill_pcc: 0.998836
first_decode_pcc: 0.998138
top_perf_samples_per_sec: 22.2438
pct_of_target: 10.0
roofline_bound: dram
optimization_level: 0
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: olmo2_1124_13b

## Test
tests/benchmark/test_llms.py::test_olmo2_1124_13b

## Model
- HF name:    allenai/OLMo-2-1124-13B
- Loader:     third_party.tt_forge_models.olmo2.causal_lm.pytorch.loader
- Variant:    OLMo_2_1124_13B

## Test config landed
- optimization_level:        0
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=1 and optimization_level=2 both fail with a compiler error:
'ttnn.scaled_dot_product_attention' op Query and result must have the same element type

## Measured (full model, defaults)
- Sample per second:  2.2175662374426244
- TTFT (ms):          2481.934794
- Prefill PCC:        0.998836
- First decode PCC:   0.998138
- Wall clock:         0:07:49
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_olmo2_1124_13b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 10.0% (2.22 / 22.24)

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
- total_flops:             848256041088
- breakdown.matmul:        848256041088
- breakdown.linear:        0
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        1677721600
- memory_bytes: 3355443200
- memory_gb:    3.125

### Params
- count:                  13716198598
- effective_count:        13202396358
- memory_bytes:           15055923988
- memory_gb:              14.021921891719103
- effective_memory_bytes: 14028319508
- effective_memory_gb:    13.064890641719103
- embedding_count:        513802240
- embedding_memory_bytes: 1027604480

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 22.2438
- top_perf_time_ms:         44.9564
- dram_time_ms:             29.9709
- compute_time_ms_lofi:     0.9639
- compute_time_ms_hifi2:    1.9279
- compute_time_ms_hifi3:    2.8918
- compute_time_ms_hifi4:    3.8557

## Files changed
- tests/benchmark/test_llms.py (added test_olmo2_1124_13b)
- tests/benchmark/benchmarks/llm_benchmark.py (general fix: hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added olmo2_1124_13b entry)

## tt-forge-models submodule
93218a34fc9fc6a671e0e41101da470c80891b2a → 73ef037570265e8b7a280ffa867a73d9a7b1130e
(submodule was already at this newer commit at session start, per hf-bringup branch)
