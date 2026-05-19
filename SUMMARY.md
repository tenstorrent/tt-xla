loader_path: third_party.tt_forge_models.mediphi.causal_lm.pytorch.loader
variant_id: Base
arch: p150
status: DONE_FAIL
test_function: test_mediphi_base
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
failure_reason: "compiler crash (Fatal Python error: Aborted) in extract_graph_helper while compiling longrope_frequency_update for Phi-3.5-based MediPhi; similar to known phi3_5_mini compiler bug (KeyError: lifted_tensor_0)"

# Benchmark added: test_mediphi_base

## Test
tests/benchmark/test_llms.py::test_mediphi_base

## Model
- HF name:    microsoft/MediPhi
- Loader:     third_party.tt_forge_models.mediphi.causal_lm.pytorch.loader
- Variant:    ModelVariant.BASE (= "Base")

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

## Failure Summary
Test fails with "Fatal Python error: Aborted" during torch.compile graph extraction
in dynamo_bridge.py's extract_graph_helper while compiling the Phi-3.5 model's
longrope_frequency_update function (transformers/modeling_rope_utils.py).

Stack trace indicates crash in:
  extract_graph_helper → extract_internal → partition_fx_graph_for_cpu_fallback
  → extract_compiled_graph_helper → extract_compiled_graph (dynamo_bridge.py)

This is the same class of compiler crash seen with test_phi3_5_mini
(# FAILED: KeyError: 'lifted_tensor_0'). MediPhi is based on Phi-3.5-mini-instruct
and inherits its LongRoPE implementation, which triggers the same compiler bug.

Infrastructure fix included: llm_benchmark.py was patched to use hasattr() guard
before calling model_loader.get_weight_dtype_config_path(), matching the pattern
in tests/runner/testers/torch/dynamic_torch_model_tester.py. This is a general
fix for loaders that don't implement this optional method.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not complete)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch: N/A
- chip_count_in_system_desc: N/A
- single_chip_assumption: N/A
- worker_grid_cores: N/A
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
- tests/benchmark/test_llms.py (added test_mediphi_base)
- tests/benchmark/benchmarks/llm_benchmark.py (hasattr guard for get_weight_dtype_config_path)

## tt-forge-models submodule
no change
