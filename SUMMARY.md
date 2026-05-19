loader_path: third_party.tt_forge_models.unsloth_phi4_gguf.causal_lm.pytorch.loader
variant_id: Mini_Instruct_Q4_K_M
arch: p150
status: DONE_FAIL
test_function: test_unsloth_phi4_mini_instruct
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
failure_reason: "compiler crash: LLVM ArrayRef<long>::back() assertion !empty() failed during longrope_frequency_update (Phi-4 LongRoPE) in dynamo bridge partition_fx_graph_for_cpu_fallback"

# Benchmark added: test_unsloth_phi4_mini_instruct

## Test
tests/benchmark/test_llms.py::test_unsloth_phi4_mini_instruct

## Model
- HF name:    unsloth/Phi-4-mini-instruct-GGUF (GGUF file: Phi-4-mini-instruct-Q4_K_M.gguf)
- Loader:     third_party.tt_forge_models.unsloth_phi4_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.PHI_4_MINI_INSTRUCT_Q4_K_M ("Mini_Instruct_Q4_K_M")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
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
Crash during num_layers=1 bring-up run (--max-output-tokens 3). The process
aborted with an LLVM assertion failure:

    python3: /opt/ttmlir-toolchain/include/llvm/ADT/ArrayRef.h:157:
    const T &llvm::ArrayRef<long>::back() const [T = long]:
    Assertion `!empty()' failed.

The crash occurs in the dynamo bridge during compilation of Phi-4's
LongRoPE frequency update (`transformers/modeling_rope_utils.py:46
longrope_frequency_update`). The call stack shows:

    extract_graph_helper
    → extract_internal
    → partition_fx_graph_for_cpu_fallback
    → extract_compiled_graph_helper
    → extract_compiled_graph
    → _call_experimental_compile

This is the same class of MLIR compiler bug seen for phi3_5_mini
(KeyError: 'lifted_tensor_0'). The fix belongs in the TT-MLIR compiler,
not the test or loader.

Additionally, a general infrastructure fix was applied: `llm_benchmark.py`
was calling `model_loader.get_weight_dtype_config_path()` unconditionally,
but the unsloth_phi4_gguf loader (and others) do not implement this method.
Fixed with a defensive `hasattr` check (matching the pattern already used
in `dynamic_torch_model_tester.py`).

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not complete)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        p150
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

## Files changed
- tests/benchmark/test_llms.py (added test_unsloth_phi4_mini_instruct)
- tests/benchmark/benchmarks/llm_benchmark.py (defensive hasattr check for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
