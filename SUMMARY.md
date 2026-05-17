loader_path: third_party.tt_forge_models.anakin87_phi_3_5_mini_ita.causal_lm.pytorch.loader
variant_id: Phi_3_5_mini_ITA
arch: n150
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
- Sample per second:  N/A (test did not complete)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150 (Wormhole n300 board, single-chip)

## Failure details
The test crashes with an LLVM assertion failure in the TT-MLIR compiler during
compilation of the decode graph:

  python3: /opt/ttmlir-toolchain/include/llvm/ADT/ArrayRef.h:157:
  const T &llvm::ArrayRef<long>::back() const [T = long]: Assertion `!empty()' failed.
  Fatal Python error: Aborted

This is an internal compiler assertion in the tt-mlir toolchain's LLVM infrastructure
(`ArrayRef::back()` called on an empty array). The crash occurs during
`extract_graph_helper` in the dynamo bridge, triggered by the Phi3ForCausalLM decode
forward pass. This is consistent with the known failures of the sibling tests
`test_phi3_mini` (TypeError: unexpected keyword arg 'cache_position') and
`test_phi3_5_mini` (KeyError: 'lifted_tensor_0'), which are also marked as FAILED
in test_llms.py — all three share the same `transformers.models.phi3.modeling_phi3`
backbone. This is a compiler-level bug not fixable in the test layer.

Additionally, during this work a general infrastructure fix was applied to
`tests/benchmark/benchmarks/llm_benchmark.py`: the `get_weight_dtype_config_path()`
call was guarded with `hasattr()` (matching the same guard already present in
`tests/runner/testers/torch/dynamic_torch_model_tester.py`), so loaders that do not
implement this optional method no longer raise AttributeError.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not produce perf_metrics JSON)

All roofline fields: N/A

## Files changed
- tests/benchmark/test_llms.py (added test_anakin87_phi_3_5_mini_ita)
- tests/benchmark/benchmarks/llm_benchmark.py (hasattr guard for get_weight_dtype_config_path)
- SUMMARY.md

## tt-forge-models submodule
no change
