loader_path: third_party.tt_forge_models.flexan_dqn_labs_dqncode_v0_3_1_2b_mlx_gguf.causal_lm.pytorch.loader
variant_id: v0.3_1.2B_MLX_Q4_K_M
arch: p150
status: DONE_FAIL
test_function: test_flexan_dqn_labs_dqncode_v0_3_1_2b_mlx_gguf
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
failure_reason: "AttributeError: 'StaticCache' object has no attribute 'conv_cache' — LFM2 model requires custom LFM2Cache with conv_cache but benchmarking harness injects StaticCache"

# Benchmark added: test_flexan_dqn_labs_dqncode_v0_3_1_2b_mlx_gguf

## Test
tests/benchmark/test_llms.py::test_flexan_dqn_labs_dqncode_v0_3_1_2b_mlx_gguf

## Model
- HF name:    Flexan/DQN-Labs-dqnCode-v0.3-1.2B-MLX-GGUF
- Loader:     third_party.tt_forge_models.flexan_dqn_labs_dqncode_v0_3_1_2b_mlx_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.DQNCODE_V0_3_1_2B_MLX_Q4_K_M

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
The test failed at CPU golden reference generation (before any TT device dispatch) with:

    AttributeError: 'StaticCache' object has no attribute 'conv_cache'

Traceback excerpt:
    transformers/models/lfm2/modeling_lfm2.py:522 in slow_forward
        past_key_values.conv_cache[self.layer_idx].copy_(conv_state)

The `lfm2` model architecture (Liquid Foundation Model 2) uses a custom cache
type that includes a `conv_cache` attribute alongside the standard KV cache.
The benchmarking harness provides a `StaticCache` (standard transformers cache)
which does not carry this attribute. This is a model/framework compatibility
issue outside the scope of this skill.

Additionally, the model load report noted a weight mismatch:
    model.layers.0.conv.conv.weight | MISMATCH | Reinit due to size mismatch -
    ckpt: torch.Size([2048, 1, 3, 1]) vs model: torch.Size([2048, 1, 3])

This suggests the GGUF conversion is producing weights with an incompatible
shape for the conv layer's 1D convolution.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A — test did not reach TT device execution
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        p150 (blackhole)
- chip_count_in_system_desc:   N/A
- single_chip_assumption:      N/A
- worker_grid_cores:           N/A
- dram_bandwidth_bytes_per_sec: N/A

### Peak FLOPs
- lofi:  N/A
- hifi2: N/A
- hifi3: N/A
- hifi4: N/A

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
