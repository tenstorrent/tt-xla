loader_path: third_party.tt_forge_models.albert_wesker_gguf.causal_lm.pytorch.loader
variant_id: 1B_i1_GGUF
arch: p150
status: DONE_FAIL
test_function: test_albert_wesker_1b_i1_gguf
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
failure_reason: "KeyError: 'sliding_attention' in transformers/models/gemma3/modeling_gemma3.py:589 — Gemma3 sliding attention architecture incompatible with installed transformers version during CPU golden generation"

# Benchmark added: test_albert_wesker_1b_i1_gguf

## Test
tests/benchmark/test_llms.py::test_albert_wesker_1b_i1_gguf

## Model
- HF name:    mradermacher/Albert_Wesker-1B-i1-GGUF
- Loader:     third_party.tt_forge_models.albert_wesker_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.ALBERT_WESKER_1B_I1_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test failed before device execution)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details
The test fails during CPU golden generation (before anything reaches TT hardware)
with `KeyError: 'sliding_attention'` in:

  venv/lib/python3.12/site-packages/transformers/models/gemma3/modeling_gemma3.py:589
    position_embeddings=position_embeddings[decoder_layer.attention_type]
  KeyError: 'sliding_attention'

The Albert Wesker 1B GGUF model uses Gemma3 architecture with sliding-window
attention. The installed transformers version does not properly populate the
`position_embeddings` dict with a `'sliding_attention'` key, causing an
immediate KeyError on the first forward call.

This is a model definition / transformers-version compatibility issue and is
not fixable within the benchmark test or infrastructure. The fix belongs in
the tt-forge-models loader (e.g. upgrading transformers or patching the GGUF
load path to map sliding_attention → global_attention).

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach device execution

## Files changed
- tests/benchmark/test_llms.py (test function added)

## tt-forge-models submodule
no change
