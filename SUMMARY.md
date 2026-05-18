loader_path: third_party.tt_forge_models.claude2_alpaca_13b_gguf.causal_lm.pytorch.loader
variant_id: 13B_GGUF
arch: p150
status: DONE_PASS
test_function: test_claude2_alpaca_13b_gguf
samples_per_second: 326.19
ttft_ms: 17.99
prefill_pcc: 0.998995
first_decode_pcc: 0.961952
top_perf_samples_per_sec: 529.3322
pct_of_target: 61.6
roofline_bound: compute
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# claude2_alpaca_13b_gguf — p150 Benchmark Summary

## Model

- **Loader**: `third_party.tt_forge_models.claude2_alpaca_13b_gguf.causal_lm.pytorch.loader`
- **Variant**: `13B_GGUF` (TheBloke/claude2-alpaca-13B-GGUF, Q4_K_M quantization)
- **HuggingFace test**: `tests/runner/test_models.py::test_all_models_torch[claude2_alpaca_13b_gguf/causal_lm/pytorch-13B_GGUF-single_device-inference]`
- **Hardware**: p150 (Blackhole, single chip)

## Test Added

`tests/benchmark/test_llms.py::test_claude2_alpaca_13b_gguf`

## Benchmark Results (Full Model, optimization_level=2, trace=True)

| Metric                  | Value       |
|-------------------------|-------------|
| Sample per second       | 326.19      |
| TTFT (ms)               | 17.99       |
| Prefill PCC             | 0.998995 ✓  |
| First decode PCC        | 0.961952 ✓  |
| Roofline (top perf)     | 529.33 s/s  |
| % of roofline target    | 61.6%       |
| Roofline bound          | compute     |

## Performance Configuration

- `optimization_level=2` (DEFAULT_OPTIMIZATION_LEVEL — SRAM optimization enabled)
- `trace_enabled=True` (default)
- `experimental_weight_dtype="bfp_bf8"` (DEFAULT_EXPERIMENTAL_WEIGHT_DTYPE)
- `batch_size=32` (default)

## Infrastructure Fix

Fixed `tests/benchmark/benchmarks/llm_benchmark.py` to guard
`get_weight_dtype_config_path()` with a `hasattr` check, matching the
pattern already used in `tests/runner/testers/torch/dynamic_torch_model_tester.py`.
This is a general fix that benefits any loader that doesn't implement this method.

## Notes

- Model is GGUF Q4_K_M quantized, dequantized to bfloat16 on CPU load (~26GB RAM)
- Test ran in 605s (10:05) on p150
- All 40 layers supported; model fits within p150's 25B single-chip capacity
- Achieved 61.6% of compute-bound roofline — gap attributable to memory bandwidth
  overhead from 13B parameter weight transfers at bfp_bf8 precision
