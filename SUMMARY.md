loader_path: third_party.tt_forge_models.davidau_q3_5_9b_opus4_6_inst_i1_gguf.causal_lm.pytorch.loader
variant_id: DavidAU_Q3_5_9B_Opus4_6_Inst_i1_Q4_K_M
arch: n150
status: DONE_FAIL
test_function: test_davidau_q3_5_9b_opus4_6_inst_i1_q4_k_m_gguf
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
failure_reason: "GGUF model architecture 'qwen35' not supported by transformers 5.2.0; ValueError raised in transformers/modeling_gguf_pytorch_utils.py before model loads"

# Benchmark added: test_davidau_q3_5_9b_opus4_6_inst_i1_q4_k_m_gguf

## Test
tests/benchmark/test_llms.py::test_davidau_q3_5_9b_opus4_6_inst_i1_q4_k_m_gguf

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
- Sample per second:  N/A (model failed to load)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150

## Failure Details

The model failed at the tokenizer/model loading step with:

```
ValueError: GGUF model with architecture qwen35 is not supported yet.
```

This error originates in `transformers/modeling_gguf_pytorch_utils.py:478` (transformers 5.2.0).
The GGUF file embeds architecture tag `qwen35`, which is not in the supported list:
`['general', 'llama', 'mistral', 'qwen2', 'qwen2_moe', 'lfm2', 'qwen3', 'qwen3_moe',
'falcon', 'tokenizer', 'phi3', 'bloom', 't5', 'stablelm', 'gpt2', 'starcoder2',
'mamba', 'nemotron', 'gemma2', 'gemma3', 'umt5', 'deci']`

This is a loader-side incompatibility (transformers library does not recognise the
`qwen35` GGUF architecture tag). It cannot be fixed by changes to the test. The fix
belongs in the tt-forge-models loader or requires a newer transformers version that
adds `qwen35` support.

## Decode roofline (first decode graph, single-chip)
N/A — model did not compile or run.

## Files changed
- tests/benchmark/test_llms.py (test function added)

## tt-forge-models submodule
no change
