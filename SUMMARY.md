loader_path: third_party.tt_forge_models.aaryank_qwen3_5_0_8b_gguf.causal_lm.pytorch.loader
variant_id: 0.8B_GGUF
arch: p150
status: DONE_FAIL
test_function: test_aaryank_qwen3_5_0_8b_gguf
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
failure_reason: "GGUF model with architecture qwen35 is not supported by transformers in venv: ValueError: GGUF model with architecture qwen35 is not supported yet."

# Benchmark added: aaryank_qwen3_5_0_8b_gguf

## Test
tests/benchmark/test_llms.py::test_aaryank_qwen3_5_0_8b_gguf

## Model
- HF name:    AaryanK/Qwen3.5-0.8B-GGUF
- Loader:     third_party.tt_forge_models.aaryank_qwen3_5_0_8b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.QWEN3_5_0_8B_GGUF (= "0.8B_GGUF")

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
- Hardware:           p150

## Failure details

The test failed at the first bring-up run (Step 3, `--num-layers 1 --max-output-tokens 3`)
with the following error from `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`:

```
ValueError: GGUF model with architecture qwen35 is not supported yet.
```

The loader calls `AutoTokenizer.from_pretrained("AaryanK/Qwen3.5-0.8B-GGUF", gguf_file="Qwen3.5-0.8B.q4_k_m.gguf")`
which triggers GGUF parsing. The GGUF metadata reports architecture `qwen35`, which the installed
`transformers` version does not yet support for GGUF loading.

This is a dependency / loader-level issue — not fixable within the benchmarking infrastructure.
The fix must land in either `transformers` (add `qwen35` GGUF support) or the loader (use a
non-GGUF model source), and belongs in the tt-forge-models repo.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A — test did not reach compilation

## Files changed
- tests/benchmark/test_llms.py (test function added)

## tt-forge-models submodule
no change
