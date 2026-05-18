loader_path: third_party.tt_forge_models.chatts_14b.causal_lm.pytorch.loader
variant_id: 14B
arch: p150
status: DONE_FAIL
test_function: test_chatts_14b
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
failure_reason: "TypeError: Qwen2TSForCausalLM.forward() got an unexpected keyword argument 'cache_position' — forward signature incompatible with transformers 5.2"

# Benchmark added: test_chatts_14b

## Test
tests/benchmark/test_llms.py::test_chatts_14b

## Model
- HF name:    bytedance-research/ChatTS-14B
- Loader:     third_party.tt_forge_models.chatts_14b.causal_lm.pytorch.loader
- Variant:    ModelVariant.CHATTS_14B (14B)

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
The test failed at CPU golden generation (before any TT device execution) with:

    TypeError: Qwen2TSForCausalLM.forward() got an unexpected keyword argument 'cache_position'

The ChatTS-14B model uses a custom `Qwen2TSForCausalLM` architecture loaded via
`trust_remote_code=True`. Its `forward()` method does not accept `cache_position`,
a keyword argument that transformers 5.2 injects during generation. This is the
same class of failure as `test_phi3_mini` (Phi3ForCausalLM.forward() cache_position).

The fix belongs in the tt-forge-models loader or in the upstream model code, not
in the benchmark harness. No files under `third_party/tt_forge_models/` were
modified.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test failed before compilation)
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
- tests/benchmark/test_llms.py (added test_chatts_14b with FAILED comment)

## tt-forge-models submodule
no change — submodule at 79cd1bb96e
