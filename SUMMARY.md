loader_path: third_party.tt_forge_models.daniloreddy_qwen3_5_9b_gguf.causal_lm.pytorch.loader
variant_id: 9B_Q4_K_M
arch: p150
status: DONE_FAIL
test_function: test_daniloreddy_qwen3_5_9b_gguf
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
failure_reason: "GGUF architecture 'qwen35' (Qwen3.5) not supported by installed transformers 5.2.0; transformers supports qwen3/qwen3_moe GGUF but not qwen35"

# Benchmark added: test_daniloreddy_qwen3_5_9b_gguf

## Test
tests/benchmark/test_llms.py::test_daniloreddy_qwen3_5_9b_gguf

## Model
- HF name:    daniloreddy/Qwen3.5-9B_GGUF
- Loader:     third_party.tt_forge_models.daniloreddy_qwen3_5_9b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.QWEN3_5_9B_Q4_K_M (= "9B_Q4_K_M")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (failed before benchmark)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details

The test fails immediately during model loading with:

```
ValueError: GGUF model with architecture qwen35 is not supported yet.
```

`transformers 5.2.0` (installed in this environment) supports GGUF loading for
`qwen3` and `qwen3_moe` architectures, but the GGUF file
`Qwen3.5-9B_Q4_K_M.gguf` declares architecture `qwen35` (Qwen 3.5), which is
not yet in `GGUF_CONFIG_MAPPING` or `GGUF_TO_TRANSFORMERS_MAPPING` for this
version. The fix requires either a newer version of transformers that adds
`qwen35` GGUF support, or a patch in the loader — both of which are out of
scope for this skill.

Architecture detected from PCI device ID `0xb140` (Blackhole = p150);
`tt-smi` was not available in the container PATH.

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach compilation/execution stage.

## Files changed
- tests/benchmark/test_llms.py (test_daniloreddy_qwen3_5_9b_gguf added with FAILED annotation)

## tt-forge-models submodule
no change
