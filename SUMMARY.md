loader_path: third_party.tt_forge_models.glm_z1_9b_0414_gguf.causal_lm.pytorch.loader
variant_id: Z1_9B_0414_GGUF
arch: p150
status: DONE_FAIL
test_function: test_glm_z1_9b_0414_gguf
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
failure_reason: "ValueError: GGUF model with architecture glm4 is not supported yet in transformers==5.2.0; AutoTokenizer.from_pretrained with gguf_file fails in loader._load_tokenizer"

# Benchmark added: test_glm_z1_9b_0414_gguf

## Test
tests/benchmark/test_llms.py::test_glm_z1_9b_0414_gguf

## Model
- HF name:    bartowski/THUDM_GLM-Z1-9B-0414-GGUF
- Loader:     third_party.tt_forge_models.glm_z1_9b_0414_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GLM_Z1_9B_0414_GGUF (value: "Z1_9B_0414_GGUF")

## Failure
The test fails before any model compilation or device execution. The transformers
library (v5.2.0) does not support the `glm4` GGUF architecture:

```
venv/lib/python3.12/site-packages/transformers/modeling_gguf_pytorch_utils.py:478:
    raise ValueError(f"GGUF model with architecture {architecture} is not supported yet.")
ValueError: GGUF model with architecture glm4 is not supported yet.
```

The error originates in `loader._load_tokenizer` which calls
`AutoTokenizer.from_pretrained(pretrained_model_name, gguf_file=THUDM_GLM-Z1-9B-0414-Q4_K_M.gguf)`.
This is inside `third_party/tt_forge_models/` and cannot be patched here.
The fix requires either upgrading transformers to a version that supports `glm4` GGUF,
or updating the loader to load the model without relying on GGUF tokenizer loading.

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (failed before execution)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (no perf metrics generated — model loading failed)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch: N/A
- chip_count_in_system_desc: N/A
- single_chip_assumption: N/A
- worker_grid_cores: N/A
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
