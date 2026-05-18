loader_path: third_party.tt_forge_models.babylm_baseline_gpt_bert_mixed.causal_lm.pytorch.loader
variant_id: 100M
arch: p150
status: DONE_FAIL
test_function: test_babylm_baseline_gpt_bert_mixed_100m
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: 2
trace_enabled: null
experimental_weight_dtype: null
failure_reason: "GPTBERTForCausalLM custom model (trust_remote_code) missing all_tied_weights_keys attribute — transformers version mismatch in custom model code: AttributeError in modeling_utils._adjust_tied_keys_with_tied_pointers"

# Benchmark added: test_babylm_baseline_gpt_bert_mixed_100m

## Test
tests/benchmark/test_llms.py::test_babylm_baseline_gpt_bert_mixed_100m

## Model
- HF name:    BabyLM-community/babylm-baseline-100m-gpt-bert-mixed
- Loader:     third_party.tt_forge_models.babylm_baseline_gpt_bert_mixed.causal_lm.pytorch.loader
- Variant:    ModelVariant.BABYLM_100M

## Test config landed
- optimization_level:        2
- trace_enabled:             true (default)
- experimental_weight_dtype: bfp_bf8 (default)
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
The test failed at model loading (before any compilation or device execution):

```
venv/lib/python3.12/site-packages/transformers/modeling_utils.py:4449:
    and not any(name in self.all_tied_weights_keys.keys() for name in names)
AttributeError: 'GPTBERTForCausalLM' object has no attribute 'all_tied_weights_keys'.
Did you mean: '_tied_weights_keys'?
```

The `GPTBERTForCausalLM` class is loaded via `trust_remote_code=True` from
`BabyLM-community/babylm-baseline-100m-gpt-bert-mixed`. The custom model
class uses the old `_tied_weights_keys` attribute, but the installed
transformers version calls `all_tied_weights_keys` in
`_adjust_tied_keys_with_tied_pointers`. This is a model-side compatibility
issue that requires updating the custom model code on HuggingFace (or pinning
transformers to an older version in the loader). No changes to the test
infrastructure can resolve this.

## Decode roofline (first decode graph, single-chip)
N/A — model never reached compilation stage.

## Files changed
- tests/benchmark/test_llms.py

## tt-forge-models submodule
no change
