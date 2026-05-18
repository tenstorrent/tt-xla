loader_path: third_party.tt_forge_models.anakin87_phi_3_5_mini_ita.causal_lm.pytorch.loader
variant_id: Phi_3_5_mini_ITA
arch: p150
status: DONE_FAIL
test_function: test_anakin87_phi_3_5_mini_ita
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
failure_reason: "compiler crash: LLVM assertion `!empty()' in ArrayRef<long>::back() during decode graph compilation (tt-mlir compiler bug)"

# Benchmark added: test_anakin87_phi_3_5_mini_ita

## Test
tests/benchmark/test_llms.py::test_anakin87_phi_3_5_mini_ita

## Model
- HF name:    anakin87/Phi-3.5-mini-ITA
- Loader:     third_party.tt_forge_models.anakin87_phi_3_5_mini_ita.causal_lm.pytorch.loader
- Variant:    ModelVariant.PHI_3_5_MINI_ITA ("Phi_3_5_mini_ITA")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (compiler crash before inference)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details
The test crashes during decode graph compilation with:
  python3: /opt/ttmlir-toolchain/include/llvm/ADT/ArrayRef.h:157:
  const T &llvm::ArrayRef<long>::back() const [T = long]: Assertion `!empty()' failed.

This is the same compiler crash previously observed on n150. The Phi-3.5 architecture
triggers an LLVM assertion in the tt-mlir compiler's ArrayRef::back() during decode
graph lowering. This is a compiler bug in tt-mlir, not fixable within this skill.

An infrastructure fix was also applied: `llm_benchmark.py` was updated to use
`hasattr(model_loader, "get_weight_dtype_config_path")` before calling the method,
matching the pattern already used in `dynamic_torch_model_tester.py`. This is a
general fix for loaders that don't implement this optional method.

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach inference stage.

## Files changed
- tests/benchmark/test_llms.py (added test_anakin87_phi_3_5_mini_ita)
- tests/benchmark/benchmarks/llm_benchmark.py (hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
