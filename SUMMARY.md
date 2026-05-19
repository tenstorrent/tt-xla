loader_path: third_party.tt_forge_models.mediphi.causal_lm.pytorch.loader
variant_id: Base
arch: p150
status: DONE_FAIL
test_function: test_mediphi
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
failure_reason: "compiler bug: LLVM ArrayRef<long>::back() assertion crash at optimization_level=1/2; ValueError Error code 13 at optimization_level=0; Phi-3.5 architecture (MediPhi is based on microsoft/Phi-3.5-mini-instruct) known to fail on TT-XLA, similar to test_phi3_mini and test_phi3_5_mini"

# Benchmark added: test_mediphi

## Test
tests/benchmark/test_llms.py::test_mediphi

## Model
- HF name:    microsoft/MediPhi
- Loader:     third_party.tt_forge_models.mediphi.causal_lm.pytorch.loader
- Variant:    ModelVariant.BASE

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test failed)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details
All optimization levels failed with compiler/runtime bugs:
- optimization_level=2: Fatal Python error (Aborted) — LLVM assertion `!empty()` in ArrayRef<long>::back() at /opt/ttmlir-toolchain/include/llvm/ADT/ArrayRef.h:157
- optimization_level=1: Same LLVM ArrayRef assertion crash
- optimization_level=0: ValueError: Error code: 13 during lm_head subgraph execution in torch_xla._XLAC._xla_sync_multi

MediPhi is based on microsoft/Phi-3.5-mini-instruct (Phi-3 architecture). This model family is known to fail on TT-XLA:
- test_phi3_mini: TypeError: Phi3ForCausalLM.forward() got unexpected kwarg 'cache_position'
- test_phi3_5_mini: KeyError: 'lifted_tensor_0'
MediPhi hits an LLVM compiler assertion at optimization_level>=1, and a device execution error (Error code 13) at optimization_level=0.

## Infrastructure fix also landed
Fixed tests/benchmark/benchmarks/llm_benchmark.py to guard `get_weight_dtype_config_path()` with `hasattr` check (matching the pattern in tests/runner/testers/torch/dynamic_torch_model_tester.py). Without this fix, any ModelLoader that does not implement `get_weight_dtype_config_path` raises AttributeError before even attempting compilation.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (model failed to compile/run)

## Files changed
- tests/benchmark/test_llms.py (added test_mediphi)
- tests/benchmark/benchmarks/llm_benchmark.py (general hasattr guard fix)

## tt-forge-models submodule
no change
