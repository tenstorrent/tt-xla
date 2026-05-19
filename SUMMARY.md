loader_path: third_party.tt_forge_models.exaone_3_5.causal_lm.pytorch.loader
variant_id: 3.5_2.4B_Instruct
arch: p150
status: DONE_FAIL
test_function: test_exaone_3_5_2_4b_instruct
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
failure_reason: "AttributeError: 'StaticCache' object has no attribute 'get_max_length' - EXAONE custom code (revision c38726a7ab4f) incompatible with transformers 5.2.0"

# Benchmark added: test_exaone_3_5_2_4b_instruct

## Test
tests/benchmark/test_llms.py::test_exaone_3_5_2_4b_instruct

## Model
- HF name:    LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct
- Loader:     third_party.tt_forge_models.exaone_3_5.causal_lm.pytorch.loader
- Variant:    ModelVariant.EXAONE_3_5_2_4B_INSTRUCT (value: "3.5_2.4B_Instruct")

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

## Failure
The EXAONE 3.5 model uses custom remote code (trust_remote_code=True, pinned to
revision c38726a7ab4f). During the CPU reference generation pass, the model's
_update_causal_mask method calls `past_key_values.get_max_length()` on the
StaticCache object, but transformers 5.2.0 renamed/removed this method
(StaticCache now has `get_max_cache_shape()` and `get_seq_length()` instead).
The error occurs before any TT device code runs.

Error:
  AttributeError: 'StaticCache' object has no attribute 'get_max_length'. Did you mean: 'get_seq_length'?
  File modeling_exaone.py:949, in _update_causal_mask

The fix must come from the EXAONE loader in tt-forge-models, either by pinning
a different model revision that is compatible with transformers 5.2.0, or by
patching the custom code compatibility layer in the loader.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not complete)
Achieved vs top_perf_samples_per_sec: N/A

## Files changed
- tests/benchmark/test_llms.py (added test_exaone_3_5_2_4b_instruct with FAILED comment)

## tt-forge-models submodule
no change (submodule at 585985bc87)
