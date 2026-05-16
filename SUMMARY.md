loader_path: third_party.tt_forge_models.aaryank_qwen3_5_4b_gguf.causal_lm.pytorch.loader
variant_id: QWEN3_5_4B_GGUF
arch: n150
status: DONE_FAIL
test_function: test_aaryank_qwen3_5_4b_gguf
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
failure_reason: "GGUF model with architecture qwen35 is not supported by transformers 5.2.0 or 5.8.1 (latest); qwen35 GGUF support is absent from all available transformers releases"

# Benchmark added: test_aaryank_qwen3_5_4b_gguf

## Test
tests/benchmark/test_llms.py::test_aaryank_qwen3_5_4b_gguf

## Model
- HF name:    AaryanK/Qwen3.5-4B-GGUF
- Loader:     third_party.tt_forge_models.aaryank_qwen3_5_4b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.QWEN3_5_4B_GGUF ("4B_GGUF")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Failure

The test was added and collected cleanly, but fails immediately during model
loading with:

```
ValueError: GGUF model with architecture qwen35 is not supported yet.
```

Raised from `transformers/modeling_gguf_pytorch_utils.py` when
`AutoTokenizer.from_pretrained("AaryanK/Qwen3.5-4B-GGUF", gguf_file="Qwen3.5-4B.q4_k_m.gguf")`
is called. The installed transformers version is 5.2.0. The latest available
version (5.8.1) was also inspected and does not include `qwen35` in its GGUF
architecture map — meaning no currently available transformers release supports
loading this GGUF checkpoint.

The loader is correct for what it is attempting to do; the fix belongs in the
transformers library (adding `qwen35` to the GGUF architecture dispatch table)
or in the tt-forge-models loader (switching to a non-GGUF loading path for
this model). No changes to `third_party/tt_forge_models/` are within scope
for this skill.

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150 (Wormhole_b0)

## Decode roofline (first decode graph, single-chip)
N/A — model did not run.

## Files changed
- tests/benchmark/test_llms.py (test function added)
- SUMMARY.md

## tt-forge-models submodule
no change — submodule remains at 6cb56d720b
