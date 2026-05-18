loader_path: third_party.tt_forge_models.davidau_q3_5_9b_opus4_6_inst_i1_gguf.causal_lm.pytorch.loader
variant_id: DavidAU_Q3_5_9B_Opus4_6_Inst_i1_Q4_K_M
arch: n150
status: DONE_FAIL
test_function: test_davidau_q3_5_9b_opus4_6_inst_i1_q4_k_m
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
failure_reason: "transformers==5.2.0 does not support qwen35 GGUF architecture: ValueError: GGUF model with architecture qwen35 is not supported yet."

# Benchmark added: test_davidau_q3_5_9b_opus4_6_inst_i1_q4_k_m

## Test
tests/benchmark/test_llms.py::test_davidau_q3_5_9b_opus4_6_inst_i1_q4_k_m

## Model
- HF name:    mradermacher/DavidAU_Q3.5_9B_Opus4.6_Inst-i1-GGUF
- Loader:     third_party.tt_forge_models.davidau_q3_5_9b_opus4_6_inst_i1_gguf.causal_lm.pytorch.loader
- Variant:    DavidAU_Q3_5_9B_Opus4_6_Inst_i1_Q4_K_M

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
The test fails immediately during model loading with:

```
ValueError: GGUF model with architecture qwen35 is not supported yet.
```

This is raised from `transformers/modeling_gguf_pytorch_utils.py:478` when trying to
load the GGUF tokenizer. The installed `transformers==5.2.0` does not have support
for the `qwen35` GGUF architecture (Qwen 3.5). The fix requires upgrading the
transformers version or adding support for `qwen35` GGUF in the loader — both
changes belong in tt-forge-models or the environment, not in this test.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A

## Files changed
- tests/benchmark/test_llms.py

## tt-forge-models submodule
no change
