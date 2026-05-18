loader_path: third_party.tt_forge_models.gemma3_12b_glm_heretic_gguf.causal_lm.pytorch.loader
variant_id: 12B_GLM_Heretic_GGUF
arch: p150
status: DONE_FAIL
test_function: test_gemma3_12b_glm_heretic_gguf
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
failure_reason: "KeyError: 'sliding_attention' in transformers gemma3 model forward (modeling_gemma3.py:589) during CPU golden generation — transformers version incompatible with GGUF model's sliding attention architecture"

# Benchmark added: test_gemma3_12b_glm_heretic_gguf

## Test
tests/benchmark/test_llms.py::test_gemma3_12b_glm_heretic_gguf

## Model
- HF name:    mradermacher/gemma-3-12b-it-vl-GLM-4.7-Flash-INSTRUCT-Thinking-Hybrid-Heretic-Uncensored-i1-GGUF
- Loader:     third_party.tt_forge_models.gemma3_12b_glm_heretic_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GEMMA_3_12B_GLM_HERETIC_GGUF

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

The test failed during CPU golden generation (before any TT device compilation)
with:

    KeyError: 'sliding_attention'

Traceback (abbreviated):
    transformers/models/gemma3/modeling_gemma3.py:589: in forward
        position_embeddings=position_embeddings[decoder_layer.attention_type],
    KeyError: 'sliding_attention'

The Gemma3 model's forward pass constructs position_embeddings keyed by attention
type, but the GGUF model reports `sliding_attention` as its attention type and no
corresponding key is present in the position_embeddings dict. This is a
transformers version / GGUF model architecture compatibility issue. The bug occurs
on the CPU reference path, meaning the model cannot be run even without TT
hardware involved.

This requires a fix in the loader (to patch the attention_type mapping) or an
upgrade/downgrade of the transformers version to one compatible with this GGUF
model's architecture. Per skill rules, loader modifications are out of scope here.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not reach TT compilation)
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
- tests/benchmark/test_llms.py (added test_gemma3_12b_glm_heretic_gguf)
- SUMMARY.md

## tt-forge-models submodule
no change
