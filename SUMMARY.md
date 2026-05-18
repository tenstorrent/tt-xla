loader_path: third_party.tt_forge_models.grok_3_reasoning_gemma3_12b_distilled_hf_gguf.causal_lm.pytorch.loader
variant_id: 12B_Q4_K_M_GGUF
arch: p150
status: DONE_FAIL
test_function: test_grok_3_reasoning_gemma3_12b_distilled_hf_gguf
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
failure_reason: "KeyError: 'sliding_attention' in transformers Gemma3 modeling_gemma3.py forward pass - transformers version incompatibility with GGUF-loaded Gemma3 12B model"

# Benchmark added: test_grok_3_reasoning_gemma3_12b_distilled_hf_gguf

## Test
tests/benchmark/test_llms.py::test_grok_3_reasoning_gemma3_12b_distilled_hf_gguf

## Model
- HF name:    mradermacher/Grok-3-reasoning-gemma3-12B-distilled-HF-GGUF
- Loader:     third_party.tt_forge_models.grok_3_reasoning_gemma3_12b_distilled_hf_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GROK_3_REASONING_GEMMA3_12B_DISTILLED_HF_Q4_K_M_GGUF (= "12B_Q4_K_M_GGUF")

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
- Wall clock:         ~2:00
- Hardware:           p150

## Failure details

The test failed during the CPU reference (prefill) forward pass — before any
device compilation or execution. The error occurs inside
`transformers/models/gemma3/modeling_gemma3.py:589`:

```
position_embeddings=position_embeddings[decoder_layer.attention_type],
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyError: 'sliding_attention'
```

The Gemma3 model loaded from the GGUF checkpoint has layers with
`attention_type == 'sliding_attention'`, but `position_embeddings` (a dict
populated upstream in the forward pass) only contains a subset of the
expected keys. This is a transformers-5.2.0 incompatibility with the
Gemma3 GGUF-loaded model variant that uses sliding-window attention layers.
The failure is in the model definition / transformers library, not in the
benchmark infrastructure or MLIR compiler. No changes to files under
`third_party/tt_forge_models/` were made.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test failed before device run)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        p150 / blackhole
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
- .github/workflows/perf-bench-matrix.json
- SUMMARY.md

## tt-forge-models submodule
no change — submodule HEAD remains at 083c0240fb
