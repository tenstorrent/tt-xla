loader_path: third_party.tt_forge_models.unsloth_phi4_gguf.causal_lm.pytorch.loader
variant_id: Mini_Instruct_Q4_K_M
arch: p150
status: DONE_FAIL
test_function: test_unsloth_phi4_mini_instruct_q4_k_m
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
failure_reason: "compiler crash: LLVM assertion !empty() in ArrayRef<long>::back() during torch.compile of Phi-4 LongRoPE positional embedding (longrope_frequency_update in modeling_phi3.py via dynamo_bridge.extract_graph_helper)"

# Benchmark added: test_unsloth_phi4_mini_instruct_q4_k_m

## Test
tests/benchmark/test_llms.py::test_unsloth_phi4_mini_instruct_q4_k_m

## Model
- HF name:    unsloth/Phi-4-mini-instruct-GGUF
- Loader:     third_party.tt_forge_models.unsloth_phi4_gguf.causal_lm.pytorch.loader
- Variant:    Mini_Instruct_Q4_K_M (ModelVariant.PHI_4_MINI_INSTRUCT_Q4_K_M)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details
The test crashed during the first torch.compile invocation (perf wrapper compilation)
with a fatal LLVM assertion error:

    python3: /opt/ttmlir-toolchain/include/llvm/ADT/ArrayRef.h:157:
    const T &llvm::ArrayRef<long>::back() const [T = long]: Assertion `!empty()' failed.

Stack trace shows the crash originates in:
    dynamo_bridge.extract_graph_helper
    → transformers/modeling_rope_utils.py:46 longrope_frequency_update
    → transformers/models/phi3/modeling_phi3.py:424 Phi3RotaryEmbedding.forward

This is a compiler bug triggered by the Phi-4 Mini LongRoPE positional embedding
computation. The LLVM ArrayRef assertion indicates an empty shape tensor is being
accessed during stablehlo graph extraction. This is outside the scope of the
benchmark test — the fix belongs in the TT-MLIR compiler.

Additionally, during the run, a general infrastructure fix was made to
`tests/benchmark/benchmarks/llm_benchmark.py`: the call to
`model_loader.get_weight_dtype_config_path()` was guarded with `hasattr()` since
not all model loaders implement this method.

## Decode roofline (first decode graph, single-chip)
N/A — test did not complete compilation

## Files changed
- tests/benchmark/test_llms.py (added test_unsloth_phi4_mini_instruct_q4_k_m)
- tests/benchmark/benchmarks/llm_benchmark.py (guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
