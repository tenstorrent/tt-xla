loader_path: third_party.tt_forge_models.gemma3_uncensored_gguf.causal_lm.pytorch.loader
variant_id: 12B_IT_UNCENSORED_GGUF
arch: p150
status: DONE_FAIL
test_function: test_gemma3_uncensored_gguf_12b_it_uncensored_gguf
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
failure_reason: "KeyError: 'sliding_attention' in transformers/models/gemma3/modeling_gemma3.py:589 — GGUF attention_type not in position_embeddings dict; CPU model forward pass fails before TT compilation"

# Benchmark added: gemma3_uncensored_gguf_12b_it_uncensored_gguf

## Test
tests/benchmark/test_llms.py::test_gemma3_uncensored_gguf_12b_it_uncensored_gguf

## Model
- HF name:    Andycurrent/gemma-3-12b-it-uncensored-GGUF
- Loader:     third_party.tt_forge_models.gemma3_uncensored_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GEMMA_3_12B_IT_UNCENSORED_GGUF (12B_IT_UNCENSORED_GGUF)

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
- Hardware:           p150 (Blackhole p300c)

## Failure
The test failed at the CPU prefill stage (before TT compilation) with:

```
KeyError: 'sliding_attention'
```

in `transformers/models/gemma3/modeling_gemma3.py:589`:
```python
position_embeddings=position_embeddings[decoder_layer.attention_type],
```

The GGUF model config sets `attention_type = 'sliding_attention'` for some
decoder layers, but the position_embeddings dict only contains keys for
`'global_attention'` (or similar). This is a compatibility issue between
the GGUF file format and the transformers `Gemma3` model implementation.
The error occurs in the CPU golden run before any TT hardware is used.

This is not fixable within this skill — the fix belongs in the transformers
library or the tt-forge-models loader.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test failed before TT compilation)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        N/A
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
- tests/benchmark/test_llms.py

## tt-forge-models submodule
no change
