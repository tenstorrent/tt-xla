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
failure_reason: "transformers==5.2.0 does not support GGUF architecture 'qwen35': ValueError: GGUF model with architecture qwen35 is not supported yet."

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

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150

## Failure Details

The test fails immediately at model loading with:

```
ValueError: GGUF model with architecture qwen35 is not supported yet.
```

Traceback:
```
transformers/modeling_gguf_pytorch_utils.py:478: in load_gguf_checkpoint
    raise ValueError(f"GGUF model with architecture {architecture} is not supported yet.")
```

The installed `transformers==5.2.0` does not support loading GGUF models that use the
`qwen35` architecture identifier (Qwen3.5 models). This is a library compatibility
issue — the loader calls `AutoTokenizer.from_pretrained` with `gguf_file=...` which
triggers GGUF parsing, and `transformers 5.2.0` only knows about a subset of GGUF
architecture strings (does not include `qwen35`).

A newer version of `transformers` is required to support this GGUF architecture type.
This is a loader/dependency fix that belongs in either the tt-forge-models repo or by
updating the pinned `transformers` version in the benchmark pyreq.

The test function has been added with a `# FAILED:` comment prefix (matching the
established pattern in `test_llms.py`) and a matrix entry added to
`.github/workflows/perf-bench-matrix.json`.

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach compilation or execution.

## Files changed
- tests/benchmark/test_llms.py (test function added with # FAILED: comment)
- .github/workflows/perf-bench-matrix.json (matrix entry added)

## tt-forge-models submodule
no change — submodule remains at 6cb56d720b
