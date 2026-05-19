loader_path: third_party.tt_forge_models.mradermacher_huihui_qwen3_5_4b_claude_4_6_opus_abliterated_i1_gguf.causal_lm.pytorch.loader
variant_id: 4B_Claude_4_6_Opus_Abliterated_i1_GGUF
arch: p150
status: DONE_FAIL
test_function: test_mradermacher_huihui_qwen3_5_4b_claude_4_6_opus_abliterated_i1_gguf
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
failure_reason: "GGUF model with architecture qwen35 is not supported in transformers 5.2.0; supported GGUF architectures include qwen3 and qwen2 but not qwen35"

# Benchmark added: test_mradermacher_huihui_qwen3_5_4b_claude_4_6_opus_abliterated_i1_gguf

## Test
tests/benchmark/test_llms.py::test_mradermacher_huihui_qwen3_5_4b_claude_4_6_opus_abliterated_i1_gguf

## Model
- HF name:    mradermacher/Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated-i1-GGUF
- Loader:     third_party.tt_forge_models.mradermacher_huihui_qwen3_5_4b_claude_4_6_opus_abliterated_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.HUIHUI_QWEN3_5_4B_CLAUDE_4_6_OPUS_ABLITERATED_I1_GGUF (value: "4B_Claude_4_6_Opus_Abliterated_i1_GGUF")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
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
The test failed during model loading with:
    ValueError: GGUF model with architecture qwen35 is not supported yet.

The GGUF file (Huihui-Qwen3.5-4B-Claude-4.6-Opus-abliterated.i1-Q4_K_M.gguf) identifies its
architecture as `qwen35`, which is not in the list of supported GGUF architectures in
transformers 5.2.0. Supported GGUF architectures are:
    ['general', 'llama', 'mistral', 'qwen2', 'qwen2_moe', 'lfm2', 'qwen3', 'qwen3_moe',
     'falcon', 'tokenizer', 'phi3', 'bloom', 't5', 'stablelm', 'gpt2', 'starcoder2', 'mamba',
     'nemotron', 'gemma2', 'gemma3', 'umt5', 'deci']

This is a loader/library incompatibility that cannot be fixed in the benchmark test code.
The fix requires either updating the transformers library to a version that supports `qwen35`,
or changing the GGUF file format in the model. This is out of scope for this skill.

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

## tt-forge-models submodule
no change
