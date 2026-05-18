loader_path: third_party.tt_forge_models.gemma_3_12b_it_ultra_uncensored_heretic_i1_gguf.causal_lm.pytorch.loader
variant_id: 12B_IT_ultra_uncensored_heretic_i1_GGUF
arch: p150
status: DONE_FAIL
test_function: test_gemma_3_12b_it_ultra_uncensored_heretic_i1_gguf
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
failure_reason: "KeyError: 'sliding_attention' in transformers/models/gemma3/modeling_gemma3.py:589 — position_embeddings dict missing 'sliding_attention' key; transformers version incompatible with Gemma3 12B sliding-attention implementation"

# Benchmark added: test_gemma_3_12b_it_ultra_uncensored_heretic_i1_gguf

## Test
tests/benchmark/test_llms.py::test_gemma_3_12b_it_ultra_uncensored_heretic_i1_gguf

## Model
- HF name:    mradermacher/gemma-3-12b-it-ultra-uncensored-heretic-i1-GGUF
- Loader:     third_party.tt_forge_models.gemma_3_12b_it_ultra_uncensored_heretic_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GEMMA_3_12B_IT_ULTRA_UNCENSORED_HERETIC_I1_GGUF

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

## Failure Details
The test failed during CPU prefill (before any TT device work) with:

    KeyError: 'sliding_attention'
    File: venv/lib/python3.12/site-packages/transformers/models/gemma3/modeling_gemma3.py:589
        position_embeddings=position_embeddings[decoder_layer.attention_type]

The Gemma3 model uses a sliding attention mechanism, and the `position_embeddings`
dictionary computed during the forward pass does not include the `'sliding_attention'`
key. This is a compatibility issue between the installed transformers version and the
Gemma3 12B GGUF model's sliding-attention implementation. The fix belongs in the
transformers library or the tt-forge-models loader — not in the benchmark test.

Tested with: --num-layers 1 --max-output-tokens 3 (optimization_level=2, trace_enabled=True)

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not reach device)
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
- tests/benchmark/test_llms.py

## tt-forge-models submodule
no change
