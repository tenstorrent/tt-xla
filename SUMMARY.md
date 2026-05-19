loader_path: third_party.tt_forge_models.lfm2_5_gguf.causal_lm.pytorch.loader
variant_id: 1_2B_Instruct_GGUF
arch: p150
status: DONE_FAIL
test_function: test_lfm2_5_1_2b_instruct_gguf
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
failure_reason: "AttributeError: 'StaticCache' object has no attribute 'conv_cache' — LFM2.5 model uses conv_cache in slow_forward (transformers/models/lfm2/modeling_lfm2.py:522) but StaticCache in this transformers version does not provide conv_cache; model-level incompatibility, not fixable in test"

# Benchmark added: test_lfm2_5_1_2b_instruct_gguf

## Test
tests/benchmark/test_llms.py::test_lfm2_5_1_2b_instruct_gguf

## Model
- HF name:    lmstudio-community/LFM2.5-1.2B-Instruct-GGUF
- Loader:     third_party.tt_forge_models.lfm2_5_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.LFM2_5_1_2B_INSTRUCT_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8" (default)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (DONE_FAIL)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details
The test fails at the first inference step with:

    AttributeError: 'StaticCache' object has no attribute 'conv_cache'

Traceback:
    venv/lib/python3.12/site-packages/transformers/models/lfm2/modeling_lfm2.py:522
    past_key_values.conv_cache[self.layer_idx].copy_(conv_state)

The LFM2.5 model (LiquidAI Foundation Model 2.5) uses a hybrid architecture
with convolutional states that require a custom cache type supporting `conv_cache`.
The benchmark harness passes `StaticCache` (standard HF KV cache), which does
not implement `conv_cache`. This is a fundamental incompatibility between the
LFM2.5 model architecture and the transformers `StaticCache` class.

Fix belongs in the tt-forge-models loader (which would need to use a custom
cache type compatible with LFM2.5's convolutional state management) or in
the transformers library adding conv_cache support to StaticCache. Neither
is within scope of this skill.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not reach inference)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch: p150 (blackhole, card type: p300c)

All roofline fields: N/A (test failed before device execution)

## Files changed
- tests/benchmark/test_llms.py (new test function test_lfm2_5_1_2b_instruct_gguf added)

## tt-forge-models submodule
no change
