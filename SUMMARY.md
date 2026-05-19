loader_path: third_party.tt_forge_models.ken3_5_9b_gguf.causal_lm.pytorch.loader
variant_id: 9B_GGUF
arch: p150
status: DONE_FAIL
test_function: test_ken3_5_9b_gguf
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
failure_reason: "GGUF model with architecture qwen35 is not supported yet in transformers 5.2.0"

# Benchmark added: test_ken3_5_9b_gguf

## Test
tests/benchmark/test_llms.py::test_ken3_5_9b_gguf

## Model
- HF name:    mradermacher/Ken3.5-9B-GGUF
- Loader:     third_party.tt_forge_models.ken3_5_9b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.KEN3_5_9B_GGUF (9B_GGUF)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details

Step 3 bring-up failed with:

```
ValueError: GGUF model with architecture qwen35 is not supported yet.
```

The `transformers` library (v5.2.0) installed in this environment does not
support the `qwen35` GGUF architecture used by the Ken3.5-9B-GGUF model. The
error occurs during `AutoTokenizer.from_pretrained(...)` with
`gguf_file="Ken3.5-9B.Q4_K_M.gguf"` inside the loader:

```
venv/lib/python3.12/site-packages/transformers/modeling_gguf_pytorch_utils.py:478:
    raise ValueError(f"GGUF model with architecture {architecture} is not supported yet.")
```

`GGUF_SUPPORTED_ARCHITECTURES` in transformers 5.2.0 includes `qwen2moe` and
`qwen3moe` but not `qwen35`. This is a library dependency issue; fixing it
requires a transformers upgrade or a custom GGUF-loading path — both out of
scope for this skill. The fix belongs in the `tt-forge-models` loader or the
environment's transformers pinning.

## Decode roofline (first decode graph, single-chip)
N/A — test did not run to completion

## Files changed
- tests/benchmark/test_llms.py (test function added, will not run successfully until transformers supports qwen35 GGUF)

## tt-forge-models submodule
no change
