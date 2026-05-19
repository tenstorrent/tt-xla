loader_path: third_party.tt_forge_models.qwen_3_5_devquasar_gguf.causal_lm.pytorch.loader
variant_id: 4B_Q4_K_M_GGUF
arch: p150
status: DONE_FAIL
test_function: test_qwen_3_5_devquasar_4b_q4_k_m_gguf
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
failure_reason: "GGUF model with architecture qwen35 is not supported yet in transformers 5.2.0: ValueError: GGUF model with architecture qwen35 is not supported yet."

# Benchmark added: test_qwen_3_5_devquasar_4b_q4_k_m_gguf

## Test
tests/benchmark/test_llms.py::test_qwen_3_5_devquasar_4b_q4_k_m_gguf

## Model
- HF name:    DevQuasar/Qwen.Qwen3.5-4B-GGUF
- Loader:     third_party.tt_forge_models.qwen_3_5_devquasar_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.QWEN_3_5_4B_Q4_K_M_GGUF (4B_Q4_K_M_GGUF)

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
The test failed at the model loading stage before any compilation or device execution.
The `transformers` library (v5.2.0) does not support loading GGUF checkpoints with
the `qwen35` architecture:

    ValueError: GGUF model with architecture qwen35 is not supported yet.

This error occurs in `transformers/modeling_gguf_pytorch_utils.py:478` when
`AutoTokenizer.from_pretrained()` tries to load the GGUF file
`Qwen.Qwen3.5-4B.Q4_K_M.gguf` from `DevQuasar/Qwen.Qwen3.5-4B-GGUF`.

The fix requires either:
1. A newer version of `transformers` that supports `qwen35` GGUF architecture, OR
2. The loader being updated to use a non-GGUF variant of the model.

This is out of scope for the benchmark skill; it belongs in tt-forge-models or
requires a transformers version upgrade.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not reach device execution)
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
- tests/benchmark/test_llms.py (added test_qwen_3_5_devquasar_4b_q4_k_m_gguf)

## tt-forge-models submodule
no change (submodule at 579493d37f)
