loader_path: third_party.tt_forge_models.baichuan.causal_lm.pytorch.loader
variant_id: 13B_Chat
arch: p150
status: DONE_FAIL
test_function: test_baichuan_13b_chat
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
failure_reason: "TypeError: 'StaticCache' object is not subscriptable - Baichuan model's custom forward code (modeling_baichuan.py from HuggingFace Hub) expects tuple-based KV cache but benchmark harness passes StaticCache (new transformers API); analogous to cache_position incompatibility, fix requires patching model definition"

# Benchmark added: test_baichuan_13b_chat

## Test
tests/benchmark/test_llms.py::test_baichuan_13b_chat

## Model
- HF name:    baichuan-inc/Baichuan-13B-Chat
- Loader:     third_party.tt_forge_models.baichuan.causal_lm.pytorch.loader
- Variant:    ModelVariant.BAICHUAN_13B_CHAT ("13B_Chat")

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

## Failure details
Test failed during CPU golden run (before TT compilation) with:

    TypeError: 'StaticCache' object is not subscriptable

The Baichuan-13B-Chat model uses custom model code downloaded from HuggingFace
Hub (`modeling_baichuan.py`). Its `forward()` method at line 307 attempts:

    past_key_values_length = past_key_values[0][0].shape[2]

The benchmark harness (`decode_utils.py`) passes a `StaticCache` object
(from `transformers.cache_utils`) as `past_key_values`. The Baichuan custom
code predates the `StaticCache` API and treats KV caches as plain tuples.
This is the same class of incompatibility as the `cache_position` issue on
`test_phi3_mini`. Fix requires patching the Baichuan model's custom forward
code in tt-forge-models or the HuggingFace Hub model code — out of scope
for this skill.

## Decode roofline (first decode graph, single-chip)
N/A — test failed before TT compilation

## Files changed
- tests/benchmark/test_llms.py (test_baichuan_13b_chat added at line 1094)

## tt-forge-models submodule
no change
