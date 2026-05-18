loader_path: third_party.tt_forge_models.bespoke_minicheck.causal_lm.pytorch.loader
variant_id: base
arch: p150
status: DONE_FAIL
test_function: test_bespoke_minicheck
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
failure_reason: "AttributeError: 'StaticCache' object has no attribute 'get_max_length' - transformers API mismatch in custom modeling_internlm2.py; get_max_length was renamed to get_seq_length in transformers 5.x"

# Benchmark added: test_bespoke_minicheck

## Test
tests/benchmark/test_llms.py::test_bespoke_minicheck

## Model
- HF name:    bespokelabs/Bespoke-MiniCheck-7B
- Loader:     third_party.tt_forge_models.bespoke_minicheck.causal_lm.pytorch.loader
- Variant:    base

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test failed)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         ~0:02:00
- Hardware:           p150

## Failure Details
The test failed with:
```
AttributeError: 'StaticCache' object has no attribute 'get_max_length'. Did you mean: 'get_seq_length'?
```

Traceback origin: `modeling_internlm2.py:1080` in `_update_causal_mask`:
```python
target_length = past_key_values.get_max_length()
```

This is a transformers API version mismatch. The `bespokelabs/Bespoke-MiniCheck-7B` model
uses a custom `modeling_internlm2.py` that calls `StaticCache.get_max_length()`, which was
renamed to `get_seq_length()` in transformers 5.x. The fix belongs in the tt-forge-models
loader or the upstream model code — this skill does not modify files under
`third_party/tt_forge_models/`.

## Decode roofline (first decode graph, single-chip)
N/A — test did not complete

## Files changed
- tests/benchmark/test_llms.py (added test_bespoke_minicheck with FAILED comment)

## tt-forge-models submodule
no change — submodule at 4d30e944dce3068c065659a12c2ce7d954a871da
