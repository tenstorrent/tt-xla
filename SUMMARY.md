loader_path: third_party.tt_forge_models.flexan_dqn_labs_dqncode_v0_3_1_2b_mlx_gguf.causal_lm.pytorch.loader
variant_id: v0.3_1.2B_MLX_Q4_K_M
arch: p150
status: DONE_FAIL
test_function: test_flexan_dqncode_v0_3_1_2b_mlx_gguf
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
failure_reason: "LFM2 hybrid architecture requires Lfm2HybridConvCache (is_compileable=False); benchmark harness passes StaticCache → AttributeError: 'StaticCache' object has no attribute 'conv_cache' during CPU reference run"

# Benchmark added: test_flexan_dqncode_v0_3_1_2b_mlx_gguf

## Test
tests/benchmark/test_llms.py::test_flexan_dqncode_v0_3_1_2b_mlx_gguf

## Model
- HF name:    Flexan/DQN-Labs-dqnCode-v0.3-1.2B-MLX-GGUF
- Loader:     third_party.tt_forge_models.flexan_dqn_labs_dqncode_v0_3_1_2b_mlx_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.DQNCODE_V0_3_1_2B_MLX_Q4_K_M (value: "v0.3_1.2B_MLX_Q4_K_M")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8" (default)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Failure details

The dqnCode model is based on the LFM2 (Liquid Foundation Model 2) architecture, which is a
hybrid model combining convolutional layers and attention layers. The transformers library
provides a specialized cache class `Lfm2HybridConvCache` for this architecture, which has both
`key_cache`/`value_cache` (for attention layers) and `conv_cache` (for convolutional layers).

`Lfm2HybridConvCache` has `is_compileable = False`, meaning it cannot be used with static
graph compilation. The benchmark harness uses `StaticCache` (from `decode_utils.init_static_cache`)
for all models, but the LFM2 model's `slow_forward` method accesses `past_key_values.conv_cache`
which does not exist on `StaticCache`.

The failure occurs during the CPU reference run (before any device compilation):

```
AttributeError: 'StaticCache' object has no attribute 'conv_cache'
  transformers/models/lfm2/modeling_lfm2.py:522: slow_forward
    past_key_values.conv_cache[self.layer_idx].copy_(conv_state)
```

To support this model class, the benchmark infrastructure would need a hybrid static cache
combining the static KV cache tensors from `StaticCache` with pre-allocated `conv_cache`
tensors, and the convolutional key_cache/value_cache would need to use in-place `copy_()`
operations instead of dynamic `torch.cat`. This is a non-trivial infrastructure change
that is out of scope for this skill.

## Measured (full model, defaults)
- Sample per second:  N/A (DONE_FAIL)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not reach compilation)
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
- tests/benchmark/test_llms.py (added test_flexan_dqncode_v0_3_1_2b_mlx_gguf with FAILED comment)

## tt-forge-models submodule
no change
