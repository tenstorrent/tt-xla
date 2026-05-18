loader_path: third_party.tt_forge_models.afrique_gemma_12b_gguf.causal_lm.pytorch.loader
variant_id: AfriqueGemma_12B_Q4_K_M
arch: p150
status: DONE_FAIL
test_function: test_afrique_gemma_12b_gguf
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
failure_reason: "KeyError: 'sliding_attention' in transformers 5.2.0 Gemma3 forward pass during CPU prefill — model-level compatibility issue"

# Benchmark added: afrique_gemma_12b_gguf

## Test
tests/benchmark/test_llms.py::test_afrique_gemma_12b_gguf

## Model
- HF name:    mradermacher/AfriqueGemma-12B-GGUF
- Loader:     third_party.tt_forge_models.afrique_gemma_12b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.AFRIQUE_GEMMA_12B_Q4_K_M

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
Test failed during bring-up at `--num-layers 1 --max-output-tokens 3` with:

    KeyError: 'sliding_attention'

The error occurs in `transformers==5.2.0` at `models/gemma3/modeling_gemma3.py:589`
during CPU prefill (before any TT device code runs):

    position_embeddings=position_embeddings[decoder_layer.attention_type],
    KeyError: 'sliding_attention'

Root cause: `position_embeddings` dict is built by iterating `self.config.layer_types`,
but `'sliding_attention'` is missing as a key when `num_hidden_layers` is overridden
to 1 and the config's `layer_types` list is truncated to only `['full_attention']` for
that single layer, while the actual first decoder layer reports `attention_type =
'sliding_attention'`. This is a transformers version compatibility issue in the Gemma3
architecture — it is not fixable by changing the test or benchmark infrastructure.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A — test did not reach compilation
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
no change — submodule at a80cdca5d5 (Fix gte_reranker_modernbert: remove return_dict=False to avoid tuple access error)
