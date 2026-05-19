loader_path: third_party.tt_forge_models.geollm_qwen3_5_9b_i1_gguf.causal_lm.pytorch.loader
variant_id: 9B_i1_GGUF
arch: p150
status: DONE_FAIL
test_function: test_geollm_qwen3_5_9b_i1_gguf
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
failure_reason: "GGUF architecture qwen35 not supported by transformers GGUF loader: ValueError: GGUF model with architecture qwen35 is not supported yet."

# Benchmark added: test_geollm_qwen3_5_9b_i1_gguf

## Test
tests/benchmark/test_llms.py::test_geollm_qwen3_5_9b_i1_gguf

## Model
- HF name:    mradermacher/GeoLLM-Qwen3.5-9B-i1-GGUF
- Loader:     third_party.tt_forge_models.geollm_qwen3_5_9b_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GEOLLM_QWEN3_5_9B_I1_GGUF (9B_i1_GGUF)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test failed before execution)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details
The test fails at model/tokenizer loading stage with:

    ValueError: GGUF model with architecture qwen35 is not supported yet.

This occurs in `transformers/modeling_gguf_pytorch_utils.py:478` inside
`load_gguf_checkpoint()`. Although transformers 5.2.0 includes a `qwen3_5`
config class, the GGUF-format loader does not yet map the `qwen35` GGUF
architecture identifier to that config class.

The same failure was observed previously for the `ken3_5_9b_i1_gguf` model
(branch `odjuricic/ai-benchmark-pipeline-test_ken3_5_9b_i1_gguf-p150`),
which uses the same transformers GGUF loading path with `qwen35` architecture.

This is a transformers library limitation — outside the scope of this skill.
The fix needs to land in transformers' GGUF architecture mapping table.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not run)
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
