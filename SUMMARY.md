loader_path: third_party.tt_forge_models.bartowski_liquidai_lfm2_2_6b_exp_gguf.causal_lm.pytorch.loader
variant_id: LIQUIDAI_LFM2_2_6B_EXP_Q4_K_M_GGUF
arch: p150
status: DONE_FAIL
test_function: test_bartowski_liquidai_lfm2_2_6b_exp_gguf
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
failure_reason: "LFM2 architecture requires Lfm2HybridConvCache (with conv_cache for hybrid conv/attention layers) but benchmark harness initializes StaticCache which lacks conv_cache attribute; AttributeError: 'StaticCache' object has no attribute 'conv_cache' in transformers/models/lfm2/modeling_lfm2.py:522"

# Benchmark added: bartowski_liquidai_lfm2_2_6b_exp_gguf

## Test
tests/benchmark/test_llms.py::test_bartowski_liquidai_lfm2_2_6b_exp_gguf

## Model
- HF name:    bartowski/LiquidAI_LFM2-2.6B-Exp-GGUF
- Loader:     third_party.tt_forge_models.bartowski_liquidai_lfm2_2_6b_exp_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.LIQUIDAI_LFM2_2_6B_EXP_Q4_K_M_GGUF

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
The LFM2-2.6B-Exp model uses a hybrid convolution/attention architecture that requires a
specialized cache class (`Lfm2HybridConvCache`) which includes both:
  - Standard KV cache (key_cache, value_cache) for attention layers
  - Convolutional state cache (conv_cache) for conv layers

The benchmark harness initializes a `StaticCache` (standard HF cache) which lacks the
`conv_cache` attribute. This causes an `AttributeError` at the very first CPU golden pass.

Error location: `transformers/models/lfm2/modeling_lfm2.py:522` in `slow_forward`:
  `past_key_values.conv_cache[self.layer_idx].copy_(conv_state)`
  `AttributeError: 'StaticCache' object has no attribute 'conv_cache'`

Supporting this would require either:
1. LFM2-specific cache initialization in the harness (model-specific, not in scope)
2. A general custom-cache-class mechanism in the harness (significant infrastructure work)

Neither is in scope for this skill per the "Things that are not ok to try" guidelines.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not reach device execution)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        p150 (Blackhole p300c)
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
- tests/benchmark/test_llms.py (added test_bartowski_liquidai_lfm2_2_6b_exp_gguf)
- .github/workflows/perf-bench-matrix.json (added bartowski_liquidai_lfm2_2_6b_exp_gguf entry)

## tt-forge-models submodule
no change
