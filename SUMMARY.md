loader_path: third_party.tt_forge_models.qwen3_5_9b_claude_4_6_opus_deckard_v4_2_uncensored_heretic_thinking_i1_gguf.causal_lm.pytorch.loader
variant_id: I1_Q4_K_M_GGUF
arch: p150
status: DONE_FAIL
test_function: test_qwen3_5_9b_claude_4_6_opus_deckard_v4_2_uncensored_heretic_thinking_i1_gguf
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
failure_reason: "GGUF model with architecture qwen35 is not supported by the current transformers version (ValueError: GGUF model with architecture qwen35 is not supported yet.) — loader-level failure, cannot be fixed in benchmark infra"

# Benchmark added: test_qwen3_5_9b_claude_4_6_opus_deckard_v4_2_uncensored_heretic_thinking_i1_gguf

## Test
tests/benchmark/test_llms.py::test_qwen3_5_9b_claude_4_6_opus_deckard_v4_2_uncensored_heretic_thinking_i1_gguf

## Model
- HF name:    mradermacher/Qwen3.5-9B-Claude-4.6-Opus-Deckard-V4.2-Uncensored-Heretic-Thinking-i1-GGUF
- Loader:     third_party.tt_forge_models.qwen3_5_9b_claude_4_6_opus_deckard_v4_2_uncensored_heretic_thinking_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.QWEN3_5_9B_CLAUDE_4_6_OPUS_DECKARD_V4_2_UNCENSORED_HERETIC_THINKING_I1_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Failure
The test failed at bring-up (num_layers=1, max_output_tokens=3) during model loading:

```
ValueError: GGUF model with architecture qwen35 is not supported yet.
```

The error originates in `transformers/modeling_gguf_pytorch_utils.py:478` when loading the tokenizer via `AutoTokenizer.from_pretrained(...)` with `gguf_file=...`. The current pinned version of `transformers` in the venv does not include support for the `qwen35` GGUF architecture. This is a loader-level / transformers-version incompatibility that cannot be fixed within the benchmark test infrastructure.

Resolution requires either:
1. Upgrading the `transformers` library to a version that supports `qwen35` GGUF, OR
2. Updating the loader to use a non-GGUF checkpoint / alternative loading strategy.

## Measured (full model, defaults)
- Sample per second:  N/A (test failed before inference)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150 (Blackhole p300c)

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach inference stage.

## Files changed
- tests/benchmark/test_llms.py (test function added, but test fails at model load)

## tt-forge-models submodule
no change
