loader_path: third_party.tt_forge_models.gemma3_1b_gguf.causal_lm.pytorch.loader
variant_id: 1B_IT_GGUF
arch: p150
status: DONE_FAIL
test_function: test_gemma3_1b_gguf
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
failure_reason: "KeyError: 'sliding_attention' in transformers Gemma3 forward pass during CPU golden reference run — position_embeddings dict missing sliding_attention key, transformers version incompatibility with GGUF model"

# Benchmark added: test_gemma3_1b_gguf

## Test
tests/benchmark/test_llms.py::test_gemma3_1b_gguf

## Model
- HF name:    ggml-org/gemma-3-1b-it-GGUF
- Loader:     third_party.tt_forge_models.gemma3_1b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GEMMA_3_1B_IT_GGUF (= "1B_IT_GGUF")

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

## Failure Details
The test failed during the CPU golden reference run (before any TT device execution) with:

    KeyError: 'sliding_attention'

at `transformers/models/gemma3/modeling_gemma3.py:589`:
    ```
    position_embeddings=position_embeddings[decoder_layer.attention_type]
    ```

The Gemma 3 1B model uses mixed attention types (global_attention + sliding_attention)
per layer. The `position_embeddings` dictionary is built without a 'sliding_attention'
key, causing a KeyError when iterating through decoder layers that use sliding attention.
This is a transformers library version incompatibility with this GGUF model at the
current installed transformers version. This is not fixable within the test or benchmark
infrastructure — the fix belongs in the transformers library or the model loader.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (no perf metrics generated — test failed before TT execution)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        p150
- chip_count_in_system_desc:   N/A
- single_chip_assumption:      N/A
- worker_grid_cores:           N/A
- dram_bandwidth_bytes_per_sec: N/A

### Roofline
- bound:                    N/A
- top_perf_samples_per_sec: N/A
- top_perf_time_ms:         N/A

## Files changed
- tests/benchmark/test_llms.py (added test_gemma3_1b_gguf)
- SUMMARY.md

## tt-forge-models submodule
no change
