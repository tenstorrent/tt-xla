loader_path: third_party.tt_forge_models.personality_assistant_9b_v1_i1_gguf.causal_lm.pytorch.loader
variant_id: 9B_V1_i1_GGUF
arch: p150
status: DONE_FAIL
test_function: test_personality_assistant_9b_v1_i1_gguf
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
failure_reason: "GGUF model with architecture qwen35 is not supported by transformers 5.2.0; GGUF_SUPPORTED_ARCHITECTURES lacks qwen35"

# Benchmark added: test_personality_assistant_9b_v1_i1_gguf

## Test
tests/benchmark/test_llms.py::test_personality_assistant_9b_v1_i1_gguf

## Model
- HF name:    mradermacher/Personality-Assistant-9B-V1-i1-GGUF
- Loader:     third_party.tt_forge_models.personality_assistant_9b_v1_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.PERSONALITY_ASSISTANT_9B_V1_I1_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test failed before inference)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure details
The bring-up run at `--num-layers 1 --max-output-tokens 3` failed immediately in
`AutoTokenizer.from_pretrained()` before any compilation or device work:

```
ValueError: GGUF model with architecture qwen35 is not supported yet.
```

The GGUF file `Personality-Assistant-9B-V1.i1-Q4_K_M.gguf` reports
`general.architecture = qwen35` (a Qwen3.5 model packaged as GGUF). The installed
transformers 5.2.0 lists the following supported GGUF architectures:

  bloom, deci, falcon, gemma2, gemma3, general, gpt2, lfm2, llama, mamba,
  mistral, nemotron, phi3, qwen2, qwen2_moe, qwen3, qwen3_moe, stablelm,
  starcoder2, t5, tokenizer, umt5

`qwen35` is absent. This is a loader / transformers-version compatibility issue —
it cannot be worked around in the test layer. The fix requires either:
1. Upgrading to a transformers version that adds `qwen35` to
   `GGUF_SUPPORTED_ARCHITECTURES`, or
2. The tt-forge-models loader switching to a non-GGUF checkpoint that uses a
   supported architecture.

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach compilation stage.

## Files changed
- tests/benchmark/test_llms.py (test function added, will remain for traceability)

## tt-forge-models submodule
no change
