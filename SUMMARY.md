loader_path: third_party.tt_forge_models.fieldmouse_ai_qwen_3_5_sovereign_vanguard_2b_gguf.causal_lm.pytorch.loader
variant_id: 2B_GGUF
arch: p150
status: DONE_FAIL
test_function: test_fieldmouse_ai_qwen_3_5_sovereign_vanguard_2b_gguf
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
failure_reason: "GGUF architecture qwen35 not supported by transformers 5.2.0: ValueError: GGUF model with architecture qwen35 is not supported yet."

# Benchmark added: test_fieldmouse_ai_qwen_3_5_sovereign_vanguard_2b_gguf

## Test
tests/benchmark/test_llms.py::test_fieldmouse_ai_qwen_3_5_sovereign_vanguard_2b_gguf

## Model
- HF name:    FieldMouse-AI/Qwen3.5-Sovereign-Vanguard-2B-GGUF
- Loader:     third_party.tt_forge_models.fieldmouse_ai_qwen_3_5_sovereign_vanguard_2b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.FIELDMOUSE_AI_QWEN_3_5_SOVEREIGN_VANGUARD_2B_GGUF (2B_GGUF)

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
- Wall clock:         N/A
- Hardware:           p150

## Failure details
The GGUF loader for this model uses the `qwen35` GGUF architecture, which is not
supported by transformers 5.2.0 (the version installed in the venv):

    ValueError: GGUF model with architecture qwen35 is not supported yet.

This occurs during tokenizer loading in `AutoTokenizer.from_pretrained()` when
reading the GGUF checkpoint. The fix requires upgrading transformers to a version
that supports `qwen35` GGUF architecture, or applying a monkey-patch to
`transformers.modeling_gguf_pytorch_utils` in the loader. Since modifying loader
files under `third_party/tt_forge_models/` is out of scope, this is recorded as
DONE_FAIL. The fix belongs in the tt-forge-models repo.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test failed before compilation)
Achieved vs top_perf_samples_per_sec: N/A

## Files changed
- tests/benchmark/test_llms.py (test function added)

## tt-forge-models submodule
no change (submodule at 593ee80213f672eb48f4e0706525da35d7243f9d)
