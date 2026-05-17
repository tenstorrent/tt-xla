loader_path: third_party.tt_forge_models.biogpt.causal_lm.pytorch.loader
variant_id: Large
arch: n150
status: DONE_FAIL
test_function: test_biogpt_large
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "compiler error: failed to legalize stablehlo.reduce (loc 'reduce.33')"

# Benchmark added: test_biogpt_large

## Test
tests/benchmark/test_llms.py::test_biogpt_large

## Model
- HF name:    microsoft/BioGPT-Large
- Loader:     third_party.tt_forge_models.biogpt.causal_lm.pytorch.loader
- Variant:    ModelVariant.BIOGPT_LARGE ("Large")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (compilation failed)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150 (Wormhole n300 L)

## Failure details

The model failed to compile at all optimization levels (`0`, `1`, `2`) with the
same error:

```
loc("reduce.33"): error: failed to legalize operation 'stablehlo.reduce'
Failed to convert from SHLO to TTIR module
ValueError: Error code: 13
```

The BioGPT model uses `stablehlo.reduce` (likely for the attention softmax or
layer-norm reduction), which the current TT-MLIR frontend cannot legalize.
This is a compiler-side issue outside the scope of this skill.

Additional bringup issue found and fixed: `llm_benchmark.py` unconditionally
called `model_loader.get_weight_dtype_config_path()` without an `hasattr`
guard, crashing for loaders that don't implement this method. The runner code
(`dynamic_torch_model_tester.py`) already had this guard; the benchmark
lacked it. Fixed to use `elif hasattr(model_loader, "get_weight_dtype_config_path"):`
in `tests/benchmark/benchmarks/llm_benchmark.py`.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (compilation failed)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        n150 (wormhole_b0)
- chip_count_in_system_desc:   N/A
- single_chip_assumption:      N/A
- worker_grid_cores:           N/A
- dram_bandwidth_bytes_per_sec: N/A

### Roofline
- bound:                    N/A
- top_perf_samples_per_sec: N/A
- top_perf_time_ms:         N/A
- dram_time_ms:             N/A
- compute_time_ms_lofi:     N/A
- compute_time_ms_hifi2:    N/A
- compute_time_ms_hifi3:    N/A
- compute_time_ms_hifi4:    N/A

## Files changed
- tests/benchmark/test_llms.py (added test_biogpt_large)
- tests/benchmark/benchmarks/llm_benchmark.py (added hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
