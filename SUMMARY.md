loader_path: third_party.tt_forge_models.chatglm.causal_lm.pytorch.loader
variant_id: 6B
arch: p150
status: DONE_FAIL
test_function: test_chatglm_6b
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
failure_reason: "ChatGLMConfig does not expose num_hidden_layers required by transformers StaticCache (uses num_layers instead); model uses non-standard config incompatible with benchmark harness cache initialization"

# Benchmark added: test_chatglm_6b

## Test
tests/benchmark/test_llms.py::test_chatglm_6b

## Model
- HF name:    zai-org/chatglm-6b
- Loader:     third_party.tt_forge_models.chatglm.causal_lm.pytorch.loader
- Variant:    ModelVariant.CHATGLM_6B (6B)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test failed)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details
The test fails at cache initialization:

    AttributeError: 'ChatGLMConfig' object has no attribute 'num_hidden_layers'

Stack trace:
  tests/benchmark/benchmarks/llm_benchmark.py:150 -> construct_inputs
  tests/benchmark/llm_utils/decode_utils.py:131 -> init_static_cache
  StaticCache(config, ...) -> transformers/cache_utils.py:1072
  -> config.num_hidden_layers fails because ChatGLMConfig uses 'num_layers' instead

The ChatGLM model uses a completely custom config object (ChatGLMConfig) with
non-standard attribute names. Transformers' StaticCache requires 'num_hidden_layers'
as part of the standard PretrainedConfig API. The fix requires either:
1. Updating the ChatGLM loader to expose num_hidden_layers (belongs in tt-forge-models repo)
2. A backwards-compatible bridge in transformers StaticCache (upstream change)

This is a model config incompatibility — not a compiler or harness bug.

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach compilation/execution phase.

## Files changed
- tests/benchmark/test_llms.py (test_chatglm_6b added)

## tt-forge-models submodule
no change — submodule at 917494c886
