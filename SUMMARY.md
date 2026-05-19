loader_path: third_party.tt_forge_models.gemma3_uncensored_gguf.causal_lm.pytorch.loader
variant_id: 12B_IT_UNCENSORED_GGUF
arch: p150
status: DONE_FAIL
test_function: test_gemma3_uncensored_gguf_12b_it_uncensored
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: null
trace_enabled: null
experimental_weight_dtype: null
failure_reason: "KeyError: 'sliding_attention' in transformers 5.2.0 modeling_gemma3.py:589 during CPU golden run — Gemma3 sliding attention position_embeddings not populated"

# Benchmark added: test_gemma3_uncensored_gguf_12b_it_uncensored

## Test
tests/benchmark/test_llms.py::test_gemma3_uncensored_gguf_12b_it_uncensored

## Model
- HF name:    Andycurrent/gemma-3-12b-it-uncensored-GGUF
- Loader:     third_party.tt_forge_models.gemma3_uncensored_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GEMMA_3_12B_IT_UNCENSORED_GGUF (value: 12B_IT_UNCENSORED_GGUF)

## Test config landed
- optimization_level:        N/A (failed before reaching device)
- trace_enabled:             N/A
- experimental_weight_dtype: N/A
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
The test failed during the CPU golden run (before reaching the TT device) with:

    KeyError: 'sliding_attention'

The error occurs in `transformers==5.2.0` at `modeling_gemma3.py:589`:

    position_embeddings=position_embeddings[decoder_layer.attention_type],

Gemma 3 has both `global_attention` and `sliding_attention` decoder layer types.
The `position_embeddings` dict is not being populated with a `sliding_attention`
key for the 12B GGUF variant when loaded via the GGUF loader path. This is a
transformers library compatibility issue with the GGUF-loaded model's configuration
and cannot be fixed in the test or benchmarking infrastructure.

tt-forge-models submodule HEAD: c9b45c4dfe

## Decode roofline (first decode graph, single-chip)
N/A — test failed before any TT device execution

## Files changed
- tests/benchmark/test_llms.py (test function added)

## tt-forge-models submodule
no change
