loader_path: third_party.tt_forge_models.harmbench_llama.causal_lm.pytorch.loader
variant_id: Llama_2_13B_cls
arch: p150
status: DONE_PASS
test_function: test_harmbench_llama_2_13b_cls
samples_per_second: 12.936
ttft_ms: 652.344
prefill_pcc: 0.998261
first_decode_pcc: 0.975254
top_perf_samples_per_sec: 22.7802
pct_of_target: 56.8
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: null

# Benchmark added: test_harmbench_llama_2_13b_cls

## Test
tests/benchmark/test_llms.py::test_harmbench_llama_2_13b_cls

## Model
- HF name:    cais/HarmBench-Llama-2-13b-cls
- Loader:     third_party.tt_forge_models.harmbench_llama.causal_lm.pytorch.loader
- Variant:    ModelVariant.HARMBENCH_LLAMA_2_13B_CLS

## Test config landed
- optimization_level:        1
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

Note: optimization_level=2 caused first decode PCC failure (~0.70); 1 is the most aggressive level
that passes PCC on the full model. Infrastructure fix: llm_benchmark.py now guards
`get_weight_dtype_config_path()` call with `hasattr()`, matching the runner's pattern.

## Measured (full model, defaults)
- Sample per second:  12.936
- TTFT (ms):          652.344
- Prefill PCC:        0.998261
- First decode PCC:   0.975254
- Wall clock:         0:05:26
- Hardware:           p150 (Blackhole p300c)

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_harmbench_llama_2_13b_cls_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 12.936 / 22.78 = 56.8%

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
- total_flops:             822503014528
- breakdown.matmul:        822503014528
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
- count:                  13015864515
- effective_count:        12852024515
- memory_bytes:           13983345416
- memory_gb:              13.023
- effective_memory_bytes: 13655665416
- effective_memory_gb:    12.718
- embedding_count:        163840000
- embedding_memory_bytes: 327680000

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 22.7802
- top_perf_time_ms:         43.8979
- dram_time_ms:             29.2652
- compute_time_ms_lofi:     0.9347
- compute_time_ms_hifi2:    1.8693
- compute_time_ms_hifi3:    2.8040
- compute_time_ms_hifi4:    3.7387

## Files changed
- tests/benchmark/test_llms.py (added test_harmbench_llama_2_13b_cls)
- tests/benchmark/benchmarks/llm_benchmark.py (hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (added harmbench_llama_2_13b_cls entry)

## tt-forge-models submodule
93218a34f → 91a1a825cb (pre-existing submodule advance, not changed by this PR)
