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
failure_reason: "GGUF architecture qwen35 not supported by transformers==5.2.0: ValueError: GGUF model with architecture qwen35 is not supported yet."

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
- experimental_weight_dtype: "bfp_bf8" (harness default)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Failure
The loader uses `AutoTokenizer.from_pretrained(..., gguf_file="Qwen3.5-4B.q4_k_m.gguf")`.
During tokenizer load, `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`
reads the GGUF file header and discovers architecture `qwen35`, which is not in the
supported-architecture table for transformers==5.2.0. The call raises:

```
ValueError: GGUF model with architecture qwen35 is not supported yet.
```

This is a library compatibility gap — transformers 5.2.0 knows `qwen3moe` and
`qwen2moe` but not `qwen35`. A newer transformers release that adds `qwen35` support
would unblock this test. The test stub has been committed with a `# FAILED:` comment
following the project convention.

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150 (Wormhole_b0)

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach the compilation stage.

## Files changed
- tests/benchmark/test_llms.py (test stub added with # FAILED comment)
- SUMMARY.md

## tt-forge-models submodule
no change (submodule HEAD: 6cb56d720b)
