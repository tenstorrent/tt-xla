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
experimental_weight_dtype: bfp_bf8
failure_reason: "GGUF model architecture 'qwen35' not supported by transformers==5.2.0; transformers 5.2.0 GGUF loader only supports qwen2/qwen3/qwen2_moe/qwen3_moe but not qwen35 (Qwen3.5)"

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

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150

## Failure details

The test fails immediately during model loading with:

```
ValueError: GGUF model with architecture qwen35 is not supported yet.
```

`transformers==5.2.0` (installed in the venv) does not support the `qwen35`
GGUF architecture. The `GGUF_SUPPORTED_ARCHITECTURES` list in
`transformers/modeling_gguf_pytorch_utils.py` includes `qwen3` and `qwen2`
but not `qwen35`, which is how llama.cpp identifies Qwen3.5 models.

The fix belongs upstream in either:
1. A newer release of `transformers` that adds `qwen35` GGUF support, or
2. The `tt-forge-models` loader rewriting the tokenizer/model loading to work
   around the missing GGUF architecture support (e.g. by loading config from
   the GGUF file and then instantiating the model a different way).

No changes were made to `third_party/tt_forge_models/`.

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach compilation or execution.

## Files changed
- tests/benchmark/test_llms.py (test function added)
- .github/workflows/perf-bench-matrix.json (matrix entry added)
- SUMMARY.md

## tt-forge-models submodule
no change
