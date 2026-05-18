loader_path: third_party.tt_forge_models.falcon_h1_tiny_r_90m_gguf.causal_lm.pytorch.loader
variant_id: Q4_K_M
arch: p150
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
experimental_weight_dtype: "bfp_bf8"
failure_reason: "hybrid Mamba+attention model: StaticCache incompatible with Mamba SSM layers (AttributeError: 'StaticCache' object has no attribute 'has_previous_state')"

# Benchmark added: falcon_h1_tiny_r_90m_gguf

## Test
tests/benchmark/test_llms.py::test_falcon_h1_tiny_r_90m_gguf

## Model
- HF name:    tiiuae/Falcon-H1-Tiny-R-90M-GGUF
- Loader:     third_party.tt_forge_models.falcon_h1_tiny_r_90m_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.FALCON_H1_TINY_R_90M_Q4_K_M (Q4_K_M)

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

The test fails at bring-up (num-layers=1, max-output-tokens=3) with:

```
AttributeError: 'StaticCache' object has no attribute 'has_previous_state'
```

Stack location:
  transformers/models/falcon_h1/modeling_falcon_h1.py:841 in torch_forward
    and cache_params.has_previous_state

Falcon-H1 is a hybrid Mamba-2 + attention model. The benchmark infrastructure
(decode_utils.py) initializes a `StaticCache` (transformers attention-only KV
cache) and passes it as `past_key_values` to the model. When the Mamba SSM
layers execute, they expect `cache_params` to be a `Mamba2Cache`/`MambaCacheParams`
object (which has `has_previous_state`), not a `StaticCache`.

This is the same class of failure as `test_mamba_2_8b` (commented out with
`# FAILED: AttributeError: 'MambaConfig' object has no attribute
'num_attention_heads'`). The benchmark infrastructure does not currently support
hybrid Mamba+attention models that require a combined cache type (e.g.
`HybridCache` or a model-specific cache factory).

Fix requires: extend `decode_utils.py` to detect hybrid SSM/attention configs
and initialize the appropriate cache type (transformers `HybridCache` or
equivalent), or add a cache_init_fn hook to `test_llm` so callers can supply
a custom cache initializer.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A
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

## Files changed
- tests/benchmark/test_llms.py (added test_falcon_h1_tiny_r_90m_gguf)

## tt-forge-models submodule
no change
