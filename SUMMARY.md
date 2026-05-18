loader_path: third_party.tt_forge_models.gemma_3_12b_character_creator_v2_gguf.causal_lm.pytorch.loader
variant_id: 12B_Character_Creator_V2_GGUF
arch: p150
status: DONE_FAIL
test_function: test_gemma_3_12b_character_creator_v2_gguf
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
failure_reason: "KeyError: 'sliding_attention' in transformers Gemma3 forward pass during CPU prefill inference — transformers library version incompatibility with model's sliding window attention mechanism"

# Benchmark added: test_gemma_3_12b_character_creator_v2_gguf

## Test
tests/benchmark/test_llms.py::test_gemma_3_12b_character_creator_v2_gguf

## Model
- HF name:    SufficientPrune3897/Gemma-3-12B-Character-Creator-V2-GGUF
- Loader:     third_party.tt_forge_models.gemma_3_12b_character_creator_v2_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GEMMA_3_12B_CHARACTER_CREATOR_V2_GGUF

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

## Failure details
The test failed during CPU prefill inference (golden reference generation) with:

    KeyError: 'sliding_attention'

Traceback location:
    transformers/models/gemma3/modeling_gemma3.py:589
    position_embeddings=position_embeddings[decoder_layer.attention_type]

The Gemma 3 12B Character Creator V2 GGUF model uses a sliding window attention
mechanism ('sliding_attention' attention type) that is not present in the
position_embeddings dict constructed by the current installed version of
transformers. This is a model/library incompatibility issue — the error occurs
before any TT device is invoked (during CPU forward pass for PCC golden).

This issue is in the transformers library and/or the model loader; no fix is
possible within the scope of this skill.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A
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
- SUMMARY.md

## tt-forge-models submodule
no change
