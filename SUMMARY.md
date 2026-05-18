loader_path: third_party.tt_forge_models.aidc_llm_laos_4b_gguf.causal_lm.pytorch.loader
variant_id: Q4_K_M
arch: p150
status: DONE_FAIL
test_function: test_aidc_llm_laos_4b_gguf
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
failure_reason: "transformers 5.2.0 Gemma3TextModel forward() KeyError: 'sliding_attention' — GGUF checkpoint layer_types include sliding_attention but position_embeddings dict missing key during CPU golden run; model incompatible with installed transformers version"

# Benchmark added: test_aidc_llm_laos_4b_gguf

## Test
tests/benchmark/test_llms.py::test_aidc_llm_laos_4b_gguf

## Model
- HF name:    mradermacher/aidc-llm-laos-4b-GGUF
- Loader:     third_party.tt_forge_models.aidc_llm_laos_4b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.AIDC_LLM_LAOS_4B_Q4_K_M (Q4_K_M)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A — DONE_FAIL
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details
The model fails during the CPU golden run (before any TT hardware compilation) with:

```
KeyError: 'sliding_attention'
```

Traceback path:
  decode_utils.py:322 → decode_utils.py:58 → gemma3/modeling_gemma3.py:653
  → gemma3/modeling_gemma3.py:589:
      position_embeddings=position_embeddings[decoder_layer.attention_type]

The Gemma3TextModel.forward() builds `position_embeddings` by iterating over
`self.config.layer_types`. The GGUF checkpoint config reports both
`sliding_attention` and `full_attention` in `layer_types`, but the actual
`position_embeddings` dict is missing the `sliding_attention` key when the
decode loop tries to access it. This is a transformers 5.2.0 compatibility
issue with Gemma3 GGUF models that use sliding window attention.

Both `--num-layers 1` and the full model reproduce the same error. The fix
belongs in the tt-forge-models loader or a transformers version upgrade.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test failed before generating perf metrics)
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
- tests/benchmark/test_llms.py (added test_aidc_llm_laos_4b_gguf)

## tt-forge-models submodule
no change
