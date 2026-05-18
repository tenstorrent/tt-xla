loader_path: third_party.tt_forge_models.granite_4_0_h_1b_gguf.causal_lm.pytorch.loader
variant_id: granite_4_0_h_1b_Q4_K_M_GGUF
arch: p150
status: DONE_FAIL
test_function: test_granite_4_0_h_1b_gguf
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
failure_reason: "GraniteMoeHybrid (Mamba-hybrid) requires HybridMambaAttentionDynamicCache (is_compileable=False); benchmark harness uses StaticCache which lacks has_previous_state; AttributeError: 'StaticCache' object has no attribute 'has_previous_state'; Mamba-hybrid architecture not supported in TT-XLA benchmark infrastructure (cf. test_mamba_2_8b)"

# Benchmark added: test_granite_4_0_h_1b_gguf

## Test
tests/benchmark/test_llms.py::test_granite_4_0_h_1b_gguf

## Model
- HF name:    ibm-granite/granite-4.0-h-1b-GGUF
- Loader:     third_party.tt_forge_models.granite_4_0_h_1b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GRANITE_4_0_H_1B_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Failure

The GraniteMoeHybrid model (`GraniteMoeHybridForCausalLM`, IBM Granite 4.0 H-1B) is a
hybrid Mamba+Attention architecture that requires `HybridMambaAttentionDynamicCache`
instead of the standard `StaticCache` used by the TT-XLA benchmark infrastructure.

`HybridMambaAttentionDynamicCache` is defined in the transformers model file with
`is_compileable = False`, meaning it cannot be traced or compiled by XLA. The Mamba
mixer layers check `cache_params.has_previous_state` (an attribute of the hybrid
cache), which `StaticCache` does not have.

Full traceback root cause:
```
transformers/models/granitemoehybrid/modeling_granitemoehybrid.py:646
    and cache_params.has_previous_state
AttributeError: 'StaticCache' object has no attribute 'has_previous_state'
```

This is the same category of failure as `test_mamba_2_8b` (commented out in
test_llms.py with `# FAILED: AttributeError: 'MambaConfig' object has no attribute
'num_attention_heads'`). Mamba-based and Mamba-hybrid models require
architecture-specific state management that is incompatible with the static
compilation + StaticCache path used by the TT-XLA benchmark harness.

**Resolution path:** Extending the benchmark harness to support Mamba-hybrid models
would require creating and managing a compatible hybrid static cache (storing Mamba
conv/SSM states as pre-allocated tensors alongside the attention KV cache) and
threading it through the XLA compilation path. This is out of scope for this skill.

## Measured (full model, defaults)
- Sample per second:  N/A (failed before compilation)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150 (4x Blackhole, PCI ID 1e52:b140)

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test failed before producing perf metrics JSON)
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
- tests/benchmark/test_llms.py (test_granite_4_0_h_1b_gguf added)
- SUMMARY.md

## tt-forge-models submodule
no change
