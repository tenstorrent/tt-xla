loader_path: third_party.tt_forge_models.qwen3_5_9b_claude_4_6_opus_deckard_v4_2_uncensored_heretic_thinking_i1_gguf.causal_lm.pytorch.loader
variant_id: I1_Q4_K_M_GGUF
arch: p150
status: DONE_FAIL
test_function: test_qwen3_5_9b_deckard_heretic_thinking_i1_gguf
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
failure_reason: "GGUF architecture 'qwen35' not supported by transformers==5.2.0 in benchmark venv (ValueError: GGUF model with architecture qwen35 is not supported yet — transformers/modeling_gguf_pytorch_utils.py:478)"

# Benchmark added: test_qwen3_5_9b_deckard_heretic_thinking_i1_gguf

## Test
tests/benchmark/test_llms.py::test_qwen3_5_9b_deckard_heretic_thinking_i1_gguf

## Model
- HF name:    mradermacher/Qwen3.5-9B-Claude-4.6-Opus-Deckard-V4.2-Uncensored-Heretic-Thinking-i1-GGUF
- Loader:     third_party.tt_forge_models.qwen3_5_9b_claude_4_6_opus_deckard_v4_2_uncensored_heretic_thinking_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.QWEN3_5_9B_CLAUDE_4_6_OPUS_DECKARD_V4_2_UNCENSORED_HERETIC_THINKING_I1_Q4_K_M_GGUF

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

## Failure
The test failed at model/tokenizer loading with:

    ValueError: GGUF model with architecture qwen35 is not supported yet.

This error is thrown by `transformers==5.2.0`'s `modeling_gguf_pytorch_utils.py:478` when
`AutoTokenizer.from_pretrained()` attempts to load the GGUF config for the
`Qwen3.5-9B-Claude-4.6-Opus-Deckard-V4.2-Uncensored-Heretic-Thinking-i1-GGUF` model.
The `qwen35` GGUF architecture is not yet supported in the pinned transformers version
used by the benchmark venv. A newer transformers release is required to load this model.
This is outside the scope of the benchmark test infrastructure to fix.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test failed before compilation)
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
- tests/benchmark/test_llms.py
- .github/workflows/perf-bench-matrix.json
- SUMMARY.md

## tt-forge-models submodule
no change
