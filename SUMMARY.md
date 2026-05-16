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
failure_reason: "GGUF model with architecture qwen35 is not supported in transformers==5.2.0; qwen35 is missing from GGUF_SUPPORTED_ARCHITECTURES"

# Benchmark added: test_aaryank_qwen3_5_4b_gguf

## Test
tests/benchmark/test_llms.py::test_aaryank_qwen3_5_4b_gguf

## Model
- HF name:    AaryanK/Qwen3.5-4B-GGUF
- Loader:     third_party.tt_forge_models.aaryank_qwen3_5_4b_gguf.causal_lm.pytorch.loader
- Variant:    QWEN3_5_4B_GGUF (value: "4B_GGUF")

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
- Hardware:           n150 (wormhole_b0)

## Failure Details

The test fails at model loading with:

```
ValueError: GGUF model with architecture qwen35 is not supported yet.
```

This is raised inside `transformers==5.2.0`'s `load_gguf_checkpoint()` at
`transformers/modeling_gguf_pytorch_utils.py:478` because `qwen35` is not
present in `GGUF_SUPPORTED_ARCHITECTURES`.

The supported Qwen GGUF architectures in `transformers==5.2.0` are:
`qwen2`, `qwen2_moe`, `qwen3`, `qwen3_moe` — but **not** `qwen35`.

The `AaryanK/Qwen3.5-4B-GGUF` model file declares `general.architecture = qwen35`,
which the current transformers version doesn't recognize. This is a
transformers-version incompatibility in the loader — no change to the test
or benchmarking infrastructure can resolve it. A newer version of `transformers`
that adds `qwen35` to its GGUF architecture mapping is required.

The `gguf` Python package itself also needed to be added as a dependency
(`pip install gguf>=0.10.0`) — the `pyreq` line in the matrix should include
it once the transformers issue is resolved.

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach compilation.

## Files changed
- tests/benchmark/test_llms.py (added test_aaryank_qwen3_5_4b_gguf with FAILED comment)

## tt-forge-models submodule
no change (submodule HEAD: 6cb56d720b)
