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
failure_reason: "GGUF architecture qwen35 not supported in transformers==5.2.0: ValueError: GGUF model with architecture qwen35 is not supported yet."

# Benchmark added: test_aaryank_qwen3_5_4b_gguf

## Test
tests/benchmark/test_llms.py::test_aaryank_qwen3_5_4b_gguf

## Model
- HF name:    AaryanK/Qwen3.5-4B-GGUF
- Loader:     third_party.tt_forge_models.aaryank_qwen3_5_4b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.QWEN3_5_4B_GGUF (value: "4B_GGUF")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Failure
The test fails immediately during tokenizer loading:

```
transformers/modeling_gguf_pytorch_utils.py:478: in load_gguf_checkpoint
    raise ValueError(f"GGUF model with architecture {architecture} is not supported yet.")
ValueError: GGUF model with architecture qwen35 is not supported yet.
```

The GGUF file `Qwen3.5-4B.q4_k_m.gguf` uses GGUF architecture `qwen35`. The
`transformers==5.2.0` GGUF loading utilities do not include a processor or
architecture mapping for `qwen35`. Supported qwen variants in 5.2.0 are
`qwen2moe` and `qwen3moe` only. The `qwen35` (Qwen 3.5) architecture would
need to be added to transformers upstream.

This cannot be resolved in this skill — modifying `third_party/tt_forge_models`
is out of scope, and bumping the transformers version is a project-level
dependency change. The fix belongs in either:
- The `tt-forge-models` repo (loader should handle missing GGUF support gracefully
  or use a non-GGUF weight path), or
- Upstream transformers (add `qwen35` GGUF architecture support).

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150 (Wormhole_b0)

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach compilation or execution phase.

## Files changed
- tests/benchmark/test_llms.py (test function added)
- .github/workflows/perf-bench-matrix.json (CI entry added)

## tt-forge-models submodule
no change
