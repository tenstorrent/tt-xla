loader_path: third_party.tt_forge_models.google_gemma3_gguf.causal_lm.pytorch.loader
variant_id: google_gemma_3_4B_IT_GGUF
arch: p150
status: DONE_FAIL
test_function: test_google_gemma3_4b_it_gguf
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
failure_reason: "model forward fails with KeyError: 'sliding_attention' in transformers 5.2.0 Gemma3 model forward (transformers/models/gemma3/modeling_gemma3.py:589 position_embeddings[decoder_layer.attention_type]); Gemma3 mixed-attention GGUF loading does not populate the sliding_attention key in position_embeddings dict"

# Benchmark added: test_google_gemma3_4b_it_gguf

## Test
tests/benchmark/test_llms.py::test_google_gemma3_4b_it_gguf

## Model
- HF name:    bartowski/google_gemma-3-4b-it-GGUF
- Loader:     third_party.tt_forge_models.google_gemma3_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GOOGLE_GEMMA_3_4B_IT_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
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

## Failure Details

The test fails at the model forward pass during both prefill (first token generation) and
the initial bring-up run with `--num-layers 1` and `--num-layers 2`.

Error location: `transformers/models/gemma3/modeling_gemma3.py:589`

```
venv/lib/python3.12/site-packages/transformers/models/gemma3/modeling_gemma3.py:589: in forward
    position_embeddings=position_embeddings[decoder_layer.attention_type],
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyError: 'sliding_attention'
```

Gemma 3 uses alternating sliding-window (local) and global attention layers. During the
model forward pass, `position_embeddings` is expected to be a dict with keys for each
attention type present in the model. The GGUF-loaded model has layers with
`attention_type == 'sliding_attention'`, but this key is absent from `position_embeddings`.

This is a model definition / transformers 5.2.0 + GGUF loading compatibility issue;
it is not a test infrastructure problem. No changes were made to the loader or the
transformers library. The fix must be applied upstream in the tt-forge-models loader
or by updating the transformers version if a fix is available.

- transformers version: 5.2.0
- Loader variant confirmed present at submodule HEAD: GOOGLE_GEMMA_3_4B_IT_GGUF

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test failed before compilation)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch: p150
- chip_count_in_system_desc: N/A
- single_chip_assumption: N/A
- worker_grid_cores: N/A
- dram_bandwidth_bytes_per_sec: N/A

### Peak FLOPs
- lofi: N/A
- hifi2: N/A
- hifi3: N/A
- hifi4: N/A

### Compute
- total_flops: N/A
- breakdown.matmul: N/A
- breakdown.linear: N/A
- breakdown.conv2d: N/A
- breakdown.sparse_matmul: N/A

### Inputs
- count: N/A
- memory_bytes: N/A

### KV cache
- count: N/A
- memory_bytes: N/A
- memory_gb: N/A

### Params
- count: N/A
- effective_count: N/A
- memory_bytes: N/A
- memory_gb: N/A
- effective_memory_bytes: N/A
- effective_memory_gb: N/A
- embedding_count: N/A
- embedding_memory_bytes: N/A

### Roofline
- bound: N/A
- top_perf_samples_per_sec: N/A
- top_perf_time_ms: N/A
- dram_time_ms: N/A
- compute_time_ms_lofi: N/A
- compute_time_ms_hifi2: N/A
- compute_time_ms_hifi3: N/A
- compute_time_ms_hifi4: N/A

## Files changed
- tests/benchmark/test_llms.py (test function added)

## tt-forge-models submodule
no change
