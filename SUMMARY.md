loader_path: third_party.tt_forge_models.gemma3n_gguf.causal_lm.pytorch.loader
variant_id: Gemma_3n_E4B_IT_Q4_K_M_GGUF
arch: p150
status: DONE_FAIL
test_function: test_gemma3n_e4b_it_q4_k_m_gguf
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
failure_reason: "loader's GGUF patch incomplete: gemma3n_text->gemma3n mapping missing in transformers get_gguf_hf_weights_map; NotImplementedError at model load"

# Benchmark added: test_gemma3n_e4b_it_q4_k_m_gguf

## Test
tests/benchmark/test_llms.py::test_gemma3n_e4b_it_q4_k_m_gguf

## Model
- HF name:    NexaAI/gemma-3n
- Loader:     third_party.tt_forge_models.gemma3n_gguf.causal_lm.pytorch.loader
- Variant:    Gemma_3n_E4B_IT_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Failure details

The model failed to load with:

```
NotImplementedError: Unknown gguf model_type: gemma3n_text in gguf-py.
```

Root cause: `transformers/modeling_gguf_pytorch_utils.py:get_gguf_hf_weights_map`
translates HuggingFace model_type names to gguf-py architecture names. It has a
mapping for `gemma3_text → gemma3` but NOT for `gemma3n_text → gemma3n`.

The loader in `third_party/tt_forge_models/gemma3n_gguf/causal_lm/pytorch/loader.py`
patches `GGUF_SUPPORTED_ARCHITECTURES` and `GGUF_TO_FAST_CONVERTERS` in transformers,
but misses patching the `get_gguf_hf_weights_map` function's model_type translation
table.

The `gguf-py` v0.19.0 does have `GEMMA3N` in `MODEL_ARCH_NAMES` (maps to `'gemma3n'`),
so the fix is: add `elif model_type == "gemma3n_text": model_type = "gemma3n"` in
transformers, OR update the loader to apply this patch.

Since neither the loader nor the `transformers` site-package can be modified in this
skill, this is recorded as DONE_FAIL. The fix belongs in the tt-forge-models loader
(patch `get_gguf_hf_weights_map`) or in upstream `transformers`.

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

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

## tt-forge-models submodule
no change
