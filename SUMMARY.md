loader_path: third_party.tt_forge_models.bartowski_liquidai_lfm2_2_6b_exp_gguf.causal_lm.pytorch.loader
variant_id: LIQUIDAI_LFM2_2_6B_EXP_Q4_K_M_GGUF
arch: n150
status: DONE_FAIL
test_function: test_liquidai_lfm2_2_6b_exp_q4_k_m_gguf
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
failure_reason: "model requires Lfm2HybridConvCache (conv_cache + KV cache for hybrid SSM/attention layers) but benchmark harness provides StaticCache; AttributeError: 'StaticCache' object has no attribute 'conv_cache' in transformers/models/lfm2/modeling_lfm2.py:522"

# Benchmark added: test_liquidai_lfm2_2_6b_exp_q4_k_m_gguf

## Test
tests/benchmark/test_llms.py::test_liquidai_lfm2_2_6b_exp_q4_k_m_gguf

## Model
- HF name:    bartowski/LiquidAI_LFM2-2.6B-Exp-GGUF
- Loader:     third_party.tt_forge_models.bartowski_liquidai_lfm2_2_6b_exp_gguf.causal_lm.pytorch.loader
- Variant:    LIQUIDAI_LFM2_2_6B_EXP_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150

## Failure Details

The LFM2-2.6B-Exp model uses a hybrid architecture combining standard
attention layers with convolutional SSM layers. The model's forward pass
requires a `Lfm2HybridConvCache` object (defined in
`transformers.models.lfm2.modeling_lfm2`) that carries both:

  - `key_cache` / `value_cache` — standard KV cache for attention layers
  - `conv_cache` — convolutional state cache for SSM layers

The benchmark harness (`llm_benchmark.py`) always initialises a
`transformers.cache_utils.StaticCache`, which only supports KV caching.
When the model's `slow_forward` tries to access
`past_key_values.conv_cache[self.layer_idx]`, it raises:

```
AttributeError: 'StaticCache' object has no attribute 'conv_cache'
  File ".../transformers/models/lfm2/modeling_lfm2.py", line 522
```

Fixing this would require either:
  (a) Adding a general "detect and instantiate the model's preferred cache
      type" mechanism to the benchmark harness — which is non-trivial since
      `Lfm2HybridConvCache` is model-specific and not a transformers
      standard base class.
  (b) Model-specific cache initialisation in the harness — explicitly
      disallowed by the skill.

Resolution: the loader or the benchmark harness needs to expose a way for
the model to declare its required cache class, so the harness can
instantiate the right type. This work belongs in the tt-forge-models or
benchmark infrastructure repos.

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach the benchmarking stage.

## Files changed
- tests/benchmark/test_llms.py (test function added; test exits with DONE_FAIL)

## tt-forge-models submodule
no change
