loader_path: third_party.tt_forge_models.anakin87_phi_3_5_mini_ita.causal_lm.pytorch.loader
variant_id: Phi_3_5_mini_ITA
arch: n150
status: DONE_FAIL
test_function: test_phi_3_5_mini_ita
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: null
failure_reason: "compiler crash: Assertion `!empty()' failed in llvm::ArrayRef::back() during longrope_frequency_update compilation (phi3 RoPE graph lowering)"

# Benchmark added: test_phi_3_5_mini_ita

## Test
tests/benchmark/test_llms.py::test_phi_3_5_mini_ita

## Model
- HF name:    anakin87/Phi-3.5-mini-ITA
- Loader:     third_party.tt_forge_models.anakin87_phi_3_5_mini_ita.causal_lm.pytorch.loader
- Variant:    ModelVariant.PHI_3_5_MINI_ITA

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: none
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (compiler crash)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n300 (n150 single-chip assumption)

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A — test did not complete
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        N/A
- chip_count_in_system_desc:   N/A
- single_chip_assumption:      N/A
- worker_grid_cores:           N/A
- dram_bandwidth_bytes_per_sec: N/A

### Peak FLOPs
- lofi:  N/A
- hifi2: N/A
- hifi3: N/A
- hifi4: N/A

### Compute
- total_flops:             N/A
- breakdown.matmul:        N/A
- breakdown.linear:        N/A
- breakdown.conv2d:        N/A
- breakdown.sparse_matmul: N/A

### Inputs
- count:        N/A
- memory_bytes: N/A

### KV cache
- count:        N/A
- memory_bytes: N/A
- memory_gb:    N/A

### Params
- count:                  N/A
- effective_count:        N/A
- memory_bytes:           N/A
- memory_gb:              N/A
- effective_memory_bytes: N/A
- effective_memory_gb:    N/A
- embedding_count:        N/A
- embedding_memory_bytes: N/A

### Roofline
- bound:                    N/A
- top_perf_samples_per_sec: N/A
- top_perf_time_ms:         N/A
- dram_time_ms:             N/A
- compute_time_ms_lofi:     N/A
- compute_time_ms_hifi2:    N/A
- compute_time_ms_hifi3:    N/A
- compute_time_ms_hifi4:    N/A

## Failure details
The test crashes with a fatal compiler assertion during the first `torch.compile` lowering pass:

```
python3: /opt/ttmlir-toolchain/include/llvm/ADT/ArrayRef.h:157:
  const T &llvm::ArrayRef<long>::back() const [T = long]: Assertion `!empty()' failed.
```

The assertion fires inside `longrope_frequency_update` (transformers/modeling_rope_utils.py), which is the dynamic LongRoPE frequency scaling used by Phi-3.5-mini architecture. The call stack at crash time is inside `extract_graph_helper` → `_call_experimental_compile`.

This is the same architecture (Phi3ForCausalLM) as the existing commented-out `test_phi3_5_mini` (which failed with a different compiler symptom: `KeyError: 'lifted_tensor_0'`). Both failures are TT-MLIR compiler bugs triggered by the Phi-3.5-mini RoPE computation graph shape.

The test has been added to `test_llms.py` with a `# FAILED:` comment matching existing convention for known-failing tests.

A general infrastructure fix was also applied: `tests/benchmark/benchmarks/llm_benchmark.py` was updated to guard the `get_weight_dtype_config_path()` call with `hasattr()`, matching the existing pattern in `tests/runner/testers/torch/dynamic_torch_model_tester.py`. This fix benefits all loaders that don't implement this optional method.

## Files changed
- tests/benchmark/test_llms.py (added test_phi_3_5_mini_ita with # FAILED comment)
- tests/benchmark/benchmarks/llm_benchmark.py (guard get_weight_dtype_config_path with hasattr)

## tt-forge-models submodule
no change
