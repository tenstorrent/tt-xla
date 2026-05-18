loader_path: third_party.tt_forge_models.falcon_h1_tiny_r_90m_gguf.causal_lm.pytorch.loader
variant_id: Q4_K_M
arch: n150
status: DONE_FAIL
test_function: test_falcon_h1_tiny_r_90m_gguf
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
failure_reason: "'StaticCache' object has no attribute 'has_previous_state' — FalconH1 requires FalconHybridMambaAttentionDynamicCache (Mamba hybrid dynamic cache, is_compileable=False), incompatible with benchmark harness StaticCache architecture"

# Benchmark added: test_falcon_h1_tiny_r_90m_gguf

## Test
tests/benchmark/test_llms.py::test_falcon_h1_tiny_r_90m_gguf

## Model
- HF name:    tiiuae/Falcon-H1-Tiny-R-90M-GGUF
- Loader:     third_party.tt_forge_models.falcon_h1_tiny_r_90m_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.FALCON_H1_TINY_R_90M_Q4_K_M (Q4_K_M)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test failed before TT device execution)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150

## Failure Details
The FalconH1 model is a hybrid Mamba-attention model that requires
`FalconHybridMambaAttentionDynamicCache` as its KV cache. This class
has `is_compileable = False` and stores Mamba SSM states (conv_states,
ssm_states) in addition to standard key/value tensors, and uses a
`has_previous_state` flag to distinguish prefill from decode steps.

The benchmark harness in `llm_benchmark.py` always initializes a
`transformers.cache_utils.StaticCache` (or MLACache) via
`construct_inputs()`. When this `StaticCache` is passed to
`FalconH1ForCausalLM.forward()`, the Mamba layer's `torch_forward()`
tries to access `cache_params.has_previous_state`, which does not exist
on `StaticCache`, raising:

    AttributeError: 'StaticCache' object has no attribute 'has_previous_state'

This fails during CPU reference generation (before any TT device
involvement). Supporting FalconH1 in the benchmark harness would require:
1. A `custom_cache_factory` path in `construct_inputs()` to create
   `FalconHybridMambaAttentionDynamicCache` instead of `StaticCache`.
2. Updated `transfer_to_device()` / `_shard_kv_cache()` to handle the
   Mamba-specific cache structure (`.conv_states` / `.ssm_states` dicts
   instead of `.layers`).
3. Handling `is_compileable = False` in the XLA trace path.

This is outside the scope of this skill. The fix belongs in the
benchmark harness with tracked issue work.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not reach TT device execution)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        n150 (wormhole_b0)
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
- tests/benchmark/test_llms.py (added test_falcon_h1_tiny_r_90m_gguf)
- SUMMARY.md

## tt-forge-models submodule
no change
