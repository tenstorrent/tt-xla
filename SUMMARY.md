loader_path: third_party.tt_forge_models.gemma_3_12b_it_vl_deepseek_v3_1_heretic_uncensored_thinking_i1_gguf.causal_lm.pytorch.loader
variant_id: 12B_IT_VL_DEEPSEEK_V3_1_I1_GGUF
arch: p150
status: DONE_FAIL
test_function: test_gemma_3_12b_it_vl_deepseek_v3_1_heretic_uncensored_thinking_i1_gguf
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
failure_reason: "model forward pass fails with KeyError: 'sliding_attention' in transformers/models/gemma3/modeling_gemma3.py:589 during CPU inference - position_embeddings dict missing 'sliding_attention' key"

# Benchmark added: test_gemma_3_12b_it_vl_deepseek_v3_1_heretic_uncensored_thinking_i1_gguf

## Test
tests/benchmark/test_llms.py::test_gemma_3_12b_it_vl_deepseek_v3_1_heretic_uncensored_thinking_i1_gguf

## Model
- HF name:    mradermacher/gemma-3-12b-it-vl-Deepseek-v3.1-Heretic-Uncensored-Thinking-i1-GGUF
- Loader:     third_party.tt_forge_models.gemma_3_12b_it_vl_deepseek_v3_1_heretic_uncensored_thinking_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GEMMA_3_12B_IT_VL_DEEPSEEK_V3_1_I1_GGUF

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
The test fails during CPU inference (the golden reference run for PCC
comparison) with:

    KeyError: 'sliding_attention'

at `transformers/models/gemma3/modeling_gemma3.py:589`:

    position_embeddings=position_embeddings[decoder_layer.attention_type],
    KeyError: 'sliding_attention'

The Gemma 3 architecture uses two attention types (global and sliding
window). The `position_embeddings` dict provided by the model's forward
pass is missing the `'sliding_attention'` key. This occurs entirely on
CPU before any TT device is involved (num_layers=1, step 3 bring-up).

This is a model definition / transformers compatibility issue. The fix
belongs in tt-forge-models or requires a compatible transformers version;
it is out of scope for this skill.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test failed before device execution)
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
