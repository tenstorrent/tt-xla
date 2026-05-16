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
optimization_level: null
trace_enabled: null
experimental_weight_dtype: null
failure_reason: "GGUF architecture 'qwen35' not supported by transformers (verified up to v5.8.1); loader cannot load the model"

# Benchmark added: test_aaryank_qwen3_5_4b_gguf

## Test
tests/benchmark/test_llms.py::test_aaryank_qwen3_5_4b_gguf

## Model
- HF name:    AaryanK/Qwen3.5-4B-GGUF
- Loader:     third_party.tt_forge_models.aaryank_qwen3_5_4b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.QWEN3_5_4B_GGUF ("4B_GGUF")

## Failure

The test fails immediately at model load time:

```
ValueError: GGUF model with architecture qwen35 is not supported yet.
```

Root cause: `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint` maintains an
allowlist (`GGUF_SUPPORTED_ARCHITECTURES`) of recognized GGUF architecture strings. The
Qwen3.5-4B GGUF file declares its architecture as `qwen35`, which is absent from the
list in every released transformers version including the latest (`5.8.1` as of 2026-05-16).

The project currently pins `transformers==5.2.0` in CI (`perf-bench-matrix.json`), and
upgrading to `5.8.1` does not resolve the issue (verified by inspecting the packaged
`modeling_gguf_pytorch_utils.py` in the `5.8.1` wheel).

## Fix required (out of scope for this skill)

Options to unblock:
1. **Loader-side**: avoid the GGUF tokenizer path and instead load the original
   (non-GGUF) Qwen3.5-4B weights + a GGUF-compatible weight quantisation step.
2. **Transformers PR**: add `qwen35` → `qwen3` or `qwen3_5` mapping in
   `GGUF_TO_TRANSFORMERS_MAPPING` inside `modeling_gguf_pytorch_utils.py` and
   release in transformers `5.9.0+`; then bump the project pin.

This is out of scope for the `add-llm-benchmark-test` skill, which is not permitted
to modify files under `third_party/tt_forge_models/` or the project's dependency pins.

## Test config landed
- optimization_level:        N/A (never reached)
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
- Hardware:           n150 (Wormhole_b0)

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not reach compilation)
Achieved vs top_perf_samples_per_sec: N/A

## Files changed
- tests/benchmark/test_llms.py  (test_aaryank_qwen3_5_4b_gguf added with # FAILED: comment)

## tt-forge-models submodule
no change
