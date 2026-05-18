loader_path: third_party.tt_forge_models.black_goo_recipe_e.causal_lm.pytorch.loader
variant_id: recipe_e
arch: n150
status: DONE_FAIL
test_function: test_black_goo_recipe_e
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: "compiler bug: TT_FATAL Invalid arguments to reshape (new_volume == old_volume assertion), RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13"

# Benchmark added: test_black_goo_recipe_e

## Test
tests/benchmark/test_llms.py::test_black_goo_recipe_e

## Model
- HF name:    KnutJaegersberg/black_goo_recipe_e
- Loader:     third_party.tt_forge_models.black_goo_recipe_e.causal_lm.pytorch.loader
- Variant:    ModelVariant.BLACK_GOO_RECIPE_E ("recipe_e")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (compiler failure)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150 (Wormhole n300 board, single chip)

## Failure Details
The test failed at `--num-layers 1 --max-output-tokens 3` during the warmup step with a compiler reshape bug:

```
TT_FATAL: Invalid arguments to reshape (assert.hpp:104)
Exception: {TT_FATAL @ reshape_common.cpp:50: new_volume == old_volume
info: Invalid arguments to reshape}
RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

The reshape mismatch occurs in the MLIR backend during compilation of the prefill/decode graph. This is a compiler-level issue outside the scope of this skill.

Additionally, the benchmarking infrastructure (`llm_benchmark.py`) called `model_loader.get_weight_dtype_config_path()` without a `hasattr` guard, causing `AttributeError` on first run. This was fixed in `llm_benchmark.py` (general fix, not model-specific), matching the existing pattern in `tests/runner/testers/torch/dynamic_torch_model_tester.py`.

## Decode roofline (first decode graph, single-chip)
N/A — compiler failure prevented any execution.

## Files changed
- tests/benchmark/test_llms.py (added test_black_goo_recipe_e)
- tests/benchmark/benchmarks/llm_benchmark.py (general infra fix: add hasattr guard for get_weight_dtype_config_path)
- SUMMARY.md

## tt-forge-models submodule
no change
