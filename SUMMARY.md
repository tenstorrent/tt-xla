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
failure_reason: "GGUF model architecture 'qwen35' not supported by any available transformers version (tested through v5.8.1); AutoTokenizer.from_pretrained raises ValueError: GGUF model with architecture qwen35 is not supported yet."

# Benchmark added: test_aaryank_qwen3_5_4b_gguf

## Test
tests/benchmark/test_llms.py::test_aaryank_qwen3_5_4b_gguf

## Model
- HF name:    AaryanK/Qwen3.5-4B-GGUF
- Loader:     third_party.tt_forge_models.aaryank_qwen3_5_4b_gguf.causal_lm.pytorch.loader
- Variant:    QWEN3_5_4B_GGUF ("4B_GGUF")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (model failed to load)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150

## Failure details

The model loader calls `AutoTokenizer.from_pretrained("AaryanK/Qwen3.5-4B-GGUF", gguf_file="Qwen3.5-4B.q4_k_m.gguf")`, which internally calls `load_gguf_checkpoint()`. The GGUF file reports architecture `qwen35`, which is not registered in transformers' `GGUF_TENSOR_MAPPING` in any published version (confirmed: transformers 5.2.0 through 5.8.1 all raise `ValueError: GGUF model with architecture qwen35 is not supported yet.`).

Full error:
```
venv/lib/python3.12/site-packages/transformers/modeling_gguf_pytorch_utils.py:478: in load_gguf_checkpoint
    raise ValueError(f"GGUF model with architecture {architecture} is not supported yet.")
ValueError: GGUF model with architecture qwen35 is not supported yet.
```

This is not fixable in the test or benchmarking infrastructure — the GGUF architecture must be registered in the transformers library. No published version of transformers supports `qwen35` GGUF as of 2026-05-16.

The fix belongs in the transformers library (or the model loader should be changed to load without GGUF format), not in tt-xla.

## Decode roofline (first decode graph, single-chip)
N/A — model never compiled.

## Files changed
- tests/benchmark/test_llms.py (test_aaryank_qwen3_5_4b_gguf added)
- .github/workflows/perf-bench-matrix.json (aaryank_qwen3_5_4b_gguf entry added)
- SUMMARY.md

## tt-forge-models submodule
no change
