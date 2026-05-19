loader_path: third_party.tt_forge_models.wizardlm_2_7b.causal_lm.pytorch.loader
variant_id: wizardlm_2_7b
arch: p150
status: DONE_FAIL
test_function: test_wizardlm_2_7b
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
failure_reason: "compiler bug: failed to legalize operation 'ttir.paged_update_cache' in TTIRToTTNNCommon pipeline for full 32-layer model (1-layer test passes at optimization_level=2)"

# Benchmark added: test_wizardlm_2_7b

## Test
tests/benchmark/test_llms.py::test_wizardlm_2_7b

## Model
- HF name:    dreamgen/WizardLM-2-7B
- Loader:     third_party.tt_forge_models.wizardlm_2_7b.causal_lm.pytorch.loader
- Variant:    ModelVariant.WIZARDLM_2_7B

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A — DONE_FAIL
- TTFT (ms):          N/A — DONE_FAIL
- Prefill PCC:        N/A — DONE_FAIL
- First decode PCC:   N/A — DONE_FAIL
- Wall clock:         N/A
- Hardware:           p150

## Failure Details
The test failed during the full model (32-layer) run with the following compiler error:

```
loc("scatter.11541"): error: failed to legalize operation 'ttir.paged_update_cache'
Failed to run TTIRToTTNNCommon pipeline
RuntimeError: Error code: 13
```

The `ttir.paged_update_cache` operation is not fully legalized for the full 32-layer
model on the p150/blackhole backend. The 1-layer test (--num-layers 1) passed cleanly
with PCC=0.999192 (prefill) and PCC=0.999515 (first decode), confirming the model
architecture is correct.

This is a TT-MLIR compiler bug (missing or incomplete lowering for
`ttir.paged_update_cache` → TTNN in the TTIRToTTNNCommon pipeline). It is out of scope
for the benchmark skill; the fix belongs in the tt-mlir compiler.

## Infrastructure fix
`tests/benchmark/benchmarks/llm_benchmark.py`: Added `hasattr` guard around
`model_loader.get_weight_dtype_config_path()` call to handle loaders that don't
implement this method (general fix, not model-specific).

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A — test did not complete
Achieved vs top_perf_samples_per_sec: N/A

## Files changed
- tests/benchmark/test_llms.py (added test_wizardlm_2_7b)
- tests/benchmark/benchmarks/llm_benchmark.py (general hasattr guard fix)

## tt-forge-models submodule
no change
