loader_path: third_party.tt_forge_models.gliese_qwen3_5_9b_abliterated_caption_gguf.causal_lm.pytorch.loader
variant_id: 9B_Abliterated_Caption_GGUF
arch: p150
status: DONE_FAIL
test_function: test_gliese_qwen3_5_9b_abliterated_caption_gguf
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
failure_reason: "GGUF model with architecture qwen35 is not supported by the installed transformers version: ValueError: GGUF model with architecture qwen35 is not supported yet."

# Benchmark added: test_gliese_qwen3_5_9b_abliterated_caption_gguf

## Test
tests/benchmark/test_llms.py::test_gliese_qwen3_5_9b_abliterated_caption_gguf

## Model
- HF name:    mradermacher/Gliese-Qwen3.5-9B-Abliterated-Caption-GGUF
- Loader:     third_party.tt_forge_models.gliese_qwen3_5_9b_abliterated_caption_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GLIESE_QWEN3_5_9B_ABLITERATED_CAPTION_GGUF

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
The test failed at the model loading stage because the installed version of the
`transformers` library does not support the `qwen35` GGUF architecture:

```
ValueError: GGUF model with architecture qwen35 is not supported yet.
```

This error occurs in `transformers/modeling_gguf_pytorch_utils.py` when
`AutoTokenizer.from_pretrained()` tries to load the GGUF checkpoint. The
loader passes `gguf_file="Gliese-Qwen3.5-9B-Abliterated-Caption.Q4_K_M.gguf"`
to `AutoTokenizer.from_pretrained()`, which in turn calls
`load_gguf_checkpoint()` — and `qwen35` is not in the supported architecture
list for the installed transformers version.

This is a transformers library compatibility issue that cannot be fixed in the
benchmark test code or benchmark infrastructure. A newer version of transformers
that supports the `qwen35` GGUF architecture is required.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not reach compilation)
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
