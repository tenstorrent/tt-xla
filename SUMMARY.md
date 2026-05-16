loader_path: third_party.tt_forge_models.albert_wesker_gguf.causal_lm.pytorch.loader
variant_id: 1B_GGUF
arch: n150
status: DONE_FAIL
test_function: test_albert_wesker_gguf_1b
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: 2
trace_enabled: null
experimental_weight_dtype: bfp_bf8
failure_reason: "ModelForwardError: KeyError: 'sliding_attention' in transformers/models/gemma3/modeling_gemma3.py:589 — GGUF-loaded Gemma3 model fails forward pass with current transformers version (position_embeddings dict missing 'sliding_attention' key)"

# Benchmark added: test_albert_wesker_gguf_1b

## Test
tests/benchmark/test_llms.py::test_albert_wesker_gguf_1b

## Model
- HF name:    mradermacher/Albert_Wesker-1B-GGUF
- Loader:     third_party.tt_forge_models.albert_wesker_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.ALBERT_WESKER_1B_GGUF (value: "1B_GGUF")

## Test config landed
- optimization_level:        2
- trace_enabled:             true (default)
- experimental_weight_dtype: "bfp_bf8" (default)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test failed before benchmark)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         ~1:14
- Hardware:           n150 (wormhole_b0, n300 single-chip context)

## Failure Details
The test failed at Step 3 (first bring-up run) with:

```
KeyError: 'sliding_attention'
  File "venv/lib/python3.12/site-packages/transformers/models/gemma3/modeling_gemma3.py", line 589, in forward
    position_embeddings=position_embeddings[decoder_layer.attention_type],
```

The model `mradermacher/Albert_Wesker-1B-GGUF` uses a Gemma3 architecture.
During the GGUF-loaded model's forward pass, `position_embeddings` does not
contain the key `'sliding_attention'` that `decoder_layer.attention_type`
maps to. This is a model/transformers version incompatibility and is NOT
fixable at the test-harness level.

Root cause is in the model's architecture configuration or a transformers
version mismatch — the loader successfully downloads and instantiates the
model but the model fails at inference time. This needs to be fixed in the
tt-forge-models loader or by upgrading/downgrading the transformers version
used for this model.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not reach compilation/execution)
Achieved vs top_perf_samples_per_sec: N/A

### System
N/A

## Files changed
- tests/benchmark/test_llms.py (added test_albert_wesker_gguf_1b)
- .github/workflows/perf-bench-matrix.json (added albert_wesker_gguf_1b entry)

## tt-forge-models submodule
no change
