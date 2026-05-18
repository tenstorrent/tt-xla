loader_path: third_party.tt_forge_models.gaiasky_qwen_3_5_gguf.causal_lm.pytorch.loader
variant_id: 9B_GGUF
arch: p150
status: DONE_FAIL
test_function: test_gaiasky_qwen_3_5_9b_gguf
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
failure_reason: "GGUF architecture 'qwen35' not supported by installed transformers version: ValueError: GGUF model with architecture qwen35 is not supported yet."

# Benchmark added: gaiasky_qwen_3_5_9b_gguf

## Test
tests/benchmark/test_llms.py::test_gaiasky_qwen_3_5_9b_gguf

## Model
- HF name:    Langurmonkey/gaiasky-qwen-3.5-gguf
- Loader:     third_party.tt_forge_models.gaiasky_qwen_3_5_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GAIASKY_QWEN_3_5_9B_GGUF

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
The model loader attempts to load the GGUF tokenizer via AutoTokenizer.from_pretrained with
gguf_file="Qwen3.5-gaiasky-9B.Q4_K_M.gguf", which calls transformers' load_gguf_checkpoint.
That function raises:

    ValueError: GGUF model with architecture qwen35 is not supported yet.

This is raised in:
    venv/lib/python3.12/site-packages/transformers/modeling_gguf_pytorch_utils.py:478

The `qwen35` GGUF architecture mapping is not present in the installed transformers version.
A newer transformers release is required to support this GGUF variant. This fix belongs in
the environment/dependency configuration, not in the loader or test.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not run)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        blackhole (p150)
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
