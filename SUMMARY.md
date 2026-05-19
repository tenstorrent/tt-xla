loader_path: third_party.tt_forge_models.lfm2_5_1_2b_thinking_kimi_v2_distill_gguf.causal_lm.pytorch.loader
variant_id: LFM2_5_1_2B_THINKING_KIMI_V2_DISTILL_Q4_K_M_GGUF
arch: p150
status: DONE_FAIL
test_function: test_lfm2_5_1_2b_thinking_kimi_v2_distill_gguf
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
failure_reason: "AttributeError: 'StaticCache' object has no attribute 'conv_cache' — LFM2 is a hybrid attention-conv model requiring Lfm2HybridConvCache (is_compileable=False); incompatible with static-cache-based benchmarking harness"

# Benchmark added: test_lfm2_5_1_2b_thinking_kimi_v2_distill_gguf

## Test
tests/benchmark/test_llms.py::test_lfm2_5_1_2b_thinking_kimi_v2_distill_gguf

## Model
- HF name:    mradermacher/LFM2.5-1.2B-Thinking-Kimi-V2-DISTILL-GGUF
- Loader:     third_party.tt_forge_models.lfm2_5_1_2b_thinking_kimi_v2_distill_gguf.causal_lm.pytorch.loader
- Variant:    LFM2_5_1_2B_THINKING_KIMI_V2_DISTILL_Q4_K_M_GGUF

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

## Failure detail
The test fails before reaching the TT device. LFM2 is a hybrid attention + short-convolution
architecture (similar to RWKV/Mamba-hybrid). Its forward pass requires `Lfm2HybridConvCache`
(defined in `transformers.models.lfm2.modeling_lfm2`) rather than `StaticCache`.

`Lfm2HybridConvCache` has `is_compileable = False` and uses dynamic key/value concatenation
(not pre-allocated static tensors). The benchmarking harness calls `init_static_cache` and
passes the resulting `StaticCache` as `past_key_values`; the model's `Lfm2ShortConv.slow_forward`
then tries to call `past_key_values.conv_cache[self.layer_idx].copy_(conv_state)`, raising:

    AttributeError: 'StaticCache' object has no attribute 'conv_cache'

Supporting this model class would require:
1. A new `init_lfm2_cache` function analogous to `init_static_cache`/`init_mla_cache`
2. Changes to `construct_inputs` and `benchmark_llm_torch_xla` to select the right cache type
3. Confirming the model can be traced/compiled with a static conv_cache (the dynamic
   key/value cache in `Lfm2HybridConvCache` may prevent static graph capture)

This is analogous to `test_mamba_2_8b` (FAILED: MambaConfig has no attribute
`num_attention_heads`) — non-transformer architectures with non-standard caches
need dedicated infrastructure support before they can be benchmarked.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A — test failed before device execution
Achieved vs top_perf_samples_per_sec: N/A

## Files changed
- tests/benchmark/test_llms.py (test function added with FAILED comment)

## tt-forge-models submodule
no change
