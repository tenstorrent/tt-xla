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
failure_reason: "GGUF model architecture 'qwen35' not supported by transformers 5.2.0 or 5.8.1 — ValueError: GGUF model with architecture qwen35 is not supported yet."

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
- experimental_weight_dtype: "bfp_bf8" (default via DEFAULT_EXPERIMENTAL_WEIGHT_DTYPE)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Failure detail

The loader calls `AutoTokenizer.from_pretrained("AaryanK/Qwen3.5-4B-GGUF", gguf_file="Qwen3.5-4B.q4_k_m.gguf")`,
which internally calls `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`. The GGUF file
embeds the architecture string `qwen35`, which has no entry in the GGUF tensor processor mapping in any
released transformers version. Both 5.2.0 (current env) and 5.8.1 (latest at time of writing) raise:

    ValueError: GGUF model with architecture qwen35 is not supported yet.

Supported qwen-family GGUF architectures in transformers 5.x: `qwen2moe`, `qwen3moe` only.

The fix requires upstream transformers to add a `qwen35` → tensor processor mapping, or the loader to
be rewritten to load weights without the GGUF path. Both are out of scope for this skill.

## Measured (full model, defaults)
- Sample per second:  N/A (model never loaded)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150 (Wormhole_b0)

## Decode roofline (first decode graph, single-chip)
N/A — loader failed before any device graph was produced.

## Files changed
- tests/benchmark/test_llms.py (test_aaryank_qwen3_5_4b_gguf at line 817, with # FAILED comment)
- .github/workflows/perf-bench-matrix.json (aaryank_qwen3_5_4b_gguf entry with gguf in pyreq)

## tt-forge-models submodule
no change
