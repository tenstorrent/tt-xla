loader_path: third_party.tt_forge_models.granite_moe_hybrid.causal_lm.pytorch.loader
variant_id: 4.0_H_350M
arch: p150
status: DONE_FAIL
test_function: test_granite_moe_hybrid_350m
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
failure_reason: "model forward pass fails during CPU golden run: 'StaticCache' object has no attribute 'has_previous_state' — GraniteMoeHybrid Mamba2 blocks require HybridMambaAttentionDynamicCache, but the harness uses StaticCache exclusively"

# Benchmark added: test_granite_moe_hybrid_350m

## Test
tests/benchmark/test_llms.py::test_granite_moe_hybrid_350m

## Model
- HF name:    ibm-granite/granite-4.0-h-350m
- Loader:     third_party.tt_forge_models.granite_moe_hybrid.causal_lm.pytorch.loader
- Variant:    ModelVariant.GRANITE_4_0_H_350M ("4.0_H_350M")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Failure details
The test fails before any TT compilation, during the CPU golden run (prefill step).

Error:
  AttributeError: 'StaticCache' object has no attribute 'has_previous_state'

Stack trace root:
  transformers/models/granitemoehybrid/modeling_granitemoehybrid.py:646:
      and cache_params.has_previous_state

The GraniteMoeHybrid architecture is a hybrid Mamba2+Attention model.
Its Mamba2 blocks access cache_params.has_previous_state, which exists
on HybridMambaAttentionDynamicCache (the cache type intended for hybrid
models) but NOT on StaticCache.

The benchmark harness (tests/benchmark/llm_utils/decode_utils.py) creates
a transformers.cache_utils.StaticCache for all models (lines 20, 120-131).
Hybrid Mamba-Attention models require a HybridMambaAttentionDynamicCache
instead. Supporting this class of models would require the harness to detect
Mamba layers and initialize the appropriate cache type.

## Measured (full model, defaults)
- Sample per second:  N/A (test did not complete)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         ~10s (failed at CPU golden run)
- Hardware:           p150

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
- tests/benchmark/test_llms.py (added test_granite_moe_hybrid_350m)
- SUMMARY.md

## tt-forge-models submodule
no change
