loader_path: third_party.tt_forge_models.gemma3_12b_glm_heretic_grande_gguf.causal_lm.pytorch.loader
variant_id: 12B_GLM_Heretic_GRANDE_GGUF
arch: p150
status: DONE_FAIL
test_function: test_gemma3_12b_glm_heretic_grande_gguf
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
failure_reason: "KeyError: 'sliding_attention' in transformers gemma3 modeling_gemma3.py:589 during CPU golden prefill - transformers version incompatibility with this GGUF variant"

# Benchmark added: test_gemma3_12b_glm_heretic_grande_gguf

## Test
tests/benchmark/test_llms.py::test_gemma3_12b_glm_heretic_grande_gguf

## Model
- HF name:    mradermacher/gemma-3-12b-it-GLM-Flash-4.7-Heretic-Thinking-Uncensored-GRANDE-i1-GGUF
- Loader:     third_party.tt_forge_models.gemma3_12b_glm_heretic_grande_gguf.causal_lm.pytorch.loader
- Variant:    GEMMA_3_12B_GLM_HERETIC_GRANDE_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test failed)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Detail

The test fails during CPU golden prefill computation (before any TT hardware involvement) with:

```
KeyError: 'sliding_attention'
  File: venv/lib/python3.12/site-packages/transformers/models/gemma3/modeling_gemma3.py:589
    position_embeddings=position_embeddings[decoder_layer.attention_type]
```

The Gemma3 GGUF model sets `decoder_layer.attention_type = 'sliding_attention'` but the
`position_embeddings` dictionary produced by `modeling_gemma3.py` does not include a
`'sliding_attention'` key. This is a transformers library version incompatibility with
this GGUF variant (the GGUF uses mixed attention types not supported by the installed
transformers version). This is not fixable in the benchmark infrastructure or test code.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not reach TT hardware)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        p150
- chip_count_in_system_desc:   N/A
- single_chip_assumption:      N/A
- worker_grid_cores:           N/A
- dram_bandwidth_bytes_per_sec: N/A

### Peak FLOPs
- lofi:  N/A
- hifi2: N/A
- hifi3: N/A
- hifi4: N/A

### Compute
- total_flops:             N/A
- breakdown.matmul:        N/A
- breakdown.linear:        N/A
- breakdown.conv2d:        N/A
- breakdown.sparse_matmul: N/A

### Inputs
- count:        N/A
- memory_bytes: N/A

### KV cache
- count:        N/A
- memory_bytes: N/A
- memory_gb:    N/A

### Params
- count:                  N/A
- effective_count:        N/A
- memory_bytes:           N/A
- memory_gb:              N/A
- effective_memory_bytes: N/A
- effective_memory_gb:    N/A
- embedding_count:        N/A
- embedding_memory_bytes: N/A

### Roofline
- bound:                    N/A
- top_perf_samples_per_sec: N/A
- top_perf_time_ms:         N/A
- dram_time_ms:             N/A
- compute_time_ms_lofi:     N/A
- compute_time_ms_hifi2:    N/A
- compute_time_ms_hifi3:    N/A
- compute_time_ms_hifi4:    N/A

## Files changed
- tests/benchmark/test_llms.py (test function added)
- SUMMARY.md

## tt-forge-models submodule
no change
