loader_path: third_party.tt_forge_models.orca_2_7b.causal_lm.pytorch.loader
variant_id: orca_2_7b
arch: p150
status: DONE_FAIL
test_function: test_orca_2_7b
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: 43.0915
pct_of_target: null
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "bfp_bf8 required for memory fit but causes Prefill PCC=0.844-0.894 at all opt levels (0,1,2), required=0.94; without bfp_bf8 OOM (Error code 12)"

# Benchmark added: test_orca_2_7b

## Test
tests/benchmark/test_llms.py::test_orca_2_7b

## Model
- HF name:    microsoft/Orca-2-7b
- Loader:     third_party.tt_forge_models.orca_2_7b.causal_lm.pytorch.loader
- Variant:    ModelVariant.ORCA_2_7B

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8 (default)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (PCC failure)
- TTFT (ms):          N/A (PCC failure)
- Prefill PCC:        0.894 (best, at opt_level=1; all opt levels fail required=0.94)
- First decode PCC:   N/A (never reached — prefill fails first)
- Wall clock:         N/A
- Hardware:           p150

## Failure analysis
All optimization levels fail PCC with experimental_weight_dtype=bfp_bf8:
- opt_level=2: Prefill PCC=0.844120 (required=0.94)
- opt_level=1: Prefill PCC=0.894365 (required=0.94)
- opt_level=0: Prefill PCC=0.859455 (required=0.94)

Without bfp_bf8 (experimental_weight_dtype=None): OOM during compilation (ValueError: Error code: 12).
bfp_bf8 is required for the 7B model to fit in p150 device memory, but causes too much
precision loss over 32 transformer layers to meet the 0.94 PCC threshold.

Infrastructure fix included: llm_benchmark.py now guards get_weight_dtype_config_path() with
hasattr() check, matching the pattern in dynamic_torch_model_tester.py, so loaders that do
not implement this method (like orca_2_7b) do not raise AttributeError.

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_orca_2_7b_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: N/A (test failed)

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
- total_flops:             425001222272
- breakdown.matmul:        425001222272
- breakdown.linear:        0
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
- count:                  6738440390
- effective_count:        6607356102
- memory_bytes:           7282734612
- memory_gb:              6.782575149089098
- effective_memory_bytes: 7020566036
- effective_memory_gb:    6.538411635905504
- embedding_count:        131084288
- embedding_memory_bytes: 262168576

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 43.0915
- top_perf_time_ms:         23.2064
- dram_time_ms:             15.4710
- compute_time_ms_lofi:     0.4830
- compute_time_ms_hifi2:    0.9659
- compute_time_ms_hifi3:    1.4489
- compute_time_ms_hifi4:    1.9318

## Files changed
- tests/benchmark/test_llms.py (added test_orca_2_7b)
- tests/benchmark/benchmarks/llm_benchmark.py (guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
