loader_path: third_party.tt_forge_models.olmo_3_7b_instruct_gguf.causal_lm.pytorch.loader
variant_id: Q4_K_M
arch: p150
status: DONE_FAIL
test_function: test_olmo_3_7b_instruct_gguf
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
failure_reason: "compiler bug: 'ttnn.scaled_dot_product_attention' op Query and result must have the same element type (MLIR error during prefill warmup with num_layers=1)"

# Benchmark added: test_olmo_3_7b_instruct_gguf

## Test
tests/benchmark/test_llms.py::test_olmo_3_7b_instruct_gguf

## Model
- HF name:    unsloth/Olmo-3-7B-Instruct-GGUF
- Loader:     third_party.tt_forge_models.olmo_3_7b_instruct_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.OLMO_3_7B_INSTRUCT_Q4_K_M (Q4_K_M)

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
- Hardware:           p150

## Failure Details
The test failed with a compiler error during the warmup (prefill) compilation step at num_layers=1:

    loc("dot.391"): error: 'ttnn.scaled_dot_product_attention' op Query and result must have the same element type
    Failed to run TTIRToTTNNCommon pipeline
    ValueError: Error code: 13

This is an MLIR/TTNN compiler bug where the attention operator expects the
query tensor and the result tensor to have the same element type, but the
OLMo 3 7B model graph produces a type mismatch. This is not fixable in the
test or benchmarking infrastructure — the fix belongs in the TT-MLIR compiler.

## Infrastructure fix
Applied a general fix to tests/benchmark/benchmarks/llm_benchmark.py: the
`get_weight_dtype_config_path()` call now uses `hasattr` guard (consistent
with tests/runner/testers/torch/dynamic_torch_model_tester.py) to gracefully
handle loaders that do not implement this optional method.

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach compile completion

## Files changed
- tests/benchmark/test_llms.py (added test_olmo_3_7b_instruct_gguf)
- tests/benchmark/benchmarks/llm_benchmark.py (hasattr guard for get_weight_dtype_config_path)
- SUMMARY.md

## tt-forge-models submodule
no change — submodule at b671ee900a
