loader_path: third_party.tt_forge_models.qwen_3_5_devquasar_gguf.causal_lm.pytorch.loader
variant_id: 4B_Q4_K_M_GGUF
arch: p150
status: DONE_FAIL
test_function: test_qwen_3_5_devquasar_gguf_4b_q4_k_m
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
failure_reason: "transformers 5.2.0 does not support GGUF architecture 'qwen35': ValueError: GGUF model with architecture qwen35 is not supported yet."

# Benchmark added: test_qwen_3_5_devquasar_gguf_4b_q4_k_m

## Test
tests/benchmark/test_llms.py::test_qwen_3_5_devquasar_gguf_4b_q4_k_m

## Model
- HF name:    DevQuasar/Qwen.Qwen3.5-4B-GGUF
- Loader:     third_party.tt_forge_models.qwen_3_5_devquasar_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.QWEN_3_5_4B_Q4_K_M_GGUF

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
The model uses the GGUF architecture string `qwen35`, which is not recognized by
transformers 5.2.0 (installed in the tt-xla venv). The supported Qwen GGUF
architectures in transformers 5.2.0 are: qwen2_moe, qwen2moe, qwen3_moe, qwen3moe.

Error:
```
ValueError: GGUF model with architecture qwen35 is not supported yet.
```

Traceback (condensed):
- `AutoTokenizer.from_pretrained(pretrained_model_name, gguf_file=...)` calls
  `load_gguf_checkpoint()` which reads the architecture key from the GGUF binary
  header and finds `qwen35` — not in the supported set.

This is not fixable by modifying the test. Remediation options:
1. Upgrade transformers to a version that supports `qwen35` GGUF.
2. Switch the loader to use the non-GGUF (bf16) checkpoint for Qwen 3.5 4B.
3. Implement `qwen35` GGUF support in transformers (upstream PR).

The test function was added to tests/benchmark/test_llms.py and can be used
once the transformers dependency is resolved.

tt-forge-models submodule HEAD at time of attempt: 579493d37f

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A — run did not reach compilation
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
- tests/benchmark/test_llms.py (test function added)

## tt-forge-models submodule
no change — submodule at 579493d37f
