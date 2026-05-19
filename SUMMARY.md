loader_path: third_party.tt_forge_models.granite_moe_hybrid.causal_lm.pytorch.loader
variant_id: 4.0_H_350M
arch: p150
status: DONE_FAIL
test_function: test_granite_moe_hybrid_4_0_h_350m
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
failure_reason: "AttributeError: StaticCache has no 'has_previous_state'; GraniteMoeHybrid requires HybridMambaAttentionDynamicCache (is_compileable=False, _can_compile_fullgraph=False due to MoE TopK gating) - incompatible with static XLA compilation harness"

# Benchmark added: test_granite_moe_hybrid_4_0_h_350m

## Test
tests/benchmark/test_llms.py::test_granite_moe_hybrid_4_0_h_350m

## Model
- HF name:    ibm-granite/granite-4.0-h-350m
- Loader:     third_party.tt_forge_models.granite_moe_hybrid.causal_lm.pytorch.loader
- Variant:    ModelVariant.GRANITE_4_0_H_350M

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

The test failed during the CPU golden run (before any TT device compilation) with:

```
AttributeError: 'StaticCache' object has no attribute 'has_previous_state'
```

Traceback location:
  `transformers/models/granitemoehybrid/modeling_granitemoehybrid.py:646` in `torch_forward`

Root cause analysis:
1. The benchmark harness always initializes `StaticCache` as `past_key_values`, but
   `GraniteMoeHybrid` is a hybrid Mamba+Attention architecture that requires
   `HybridMambaAttentionDynamicCache` (model-internal class defined in the transformers
   modeling file).
2. `HybridMambaAttentionDynamicCache.is_compileable = False` — the cache itself is
   marked non-compileable, making it incompatible with TT XLA's static graph approach.
3. `GraniteMoeHybridModel._can_compile_fullgraph = False` — TopK gating in the MoE
   layers prevents full-graph tracing (noted in the transformers source:
   "TopK gating fails fullgraph compilation at 'expert_size = expert_size.tolist()'").
4. Mamba SSM layers (28/32 layers are mamba) have state-dependent computation that
   does not reduce to a static graph.

Conclusion: This model is architecturally incompatible with the TT XLA static
compilation harness. The failure is not fixable by adjusting test parameters.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not reach TT compilation)
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
- tests/benchmark/test_llms.py

## tt-forge-models submodule
no change
