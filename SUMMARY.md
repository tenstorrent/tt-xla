loader_path: third_party.tt_forge_models.mradermacher_qwen3_5_4b_claude_4_6_opus_reasoning_distill_heretic_v3_i1_gguf.causal_lm.pytorch.loader
variant_id: 4B_Claude_4.6_Opus_Reasoning_Distill_heretic_v3_i1_GGUF
arch: p150
status: DONE_FAIL
test_function: test_mradermacher_qwen3_5_4b_claude_4_6_opus_reasoning_distill_heretic_v3_i1_gguf
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
failure_reason: "loader bug: _patch_qwen35_support() does not add 'qwen35' to transformers CONFIG_MAPPING; AutoTokenizer.from_pretrained fails with ValueError: Unrecognized model identifier: qwen35"

# Benchmark added: test_mradermacher_qwen3_5_4b_claude_4_6_opus_reasoning_distill_heretic_v3_i1_gguf

## Test
tests/benchmark/test_llms.py::test_mradermacher_qwen3_5_4b_claude_4_6_opus_reasoning_distill_heretic_v3_i1_gguf

## Model
- HF name:    mradermacher/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distill-heretic-v3-i1-GGUF
- Loader:     third_party.tt_forge_models.mradermacher_qwen3_5_4b_claude_4_6_opus_reasoning_distill_heretic_v3_i1_gguf.causal_lm.pytorch.loader
- Variant:    4B_Claude_4.6_Opus_Reasoning_Distill_heretic_v3_i1_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Failure
The bring-up test (`--num-layers 1 --max-output-tokens 3`) failed during tokenizer
loading before any compilation or device work occurred.

Error in `loader.py:_load_tokenizer` →
`AutoTokenizer.from_pretrained(pretrained_model_name, gguf_file=GGUF_FILE)`:

```
ValueError: Unrecognized model identifier: qwen35. Should contain one of ...
```

The loader's `_patch_qwen35_support()` adds `qwen35` to
`GGUF_SUPPORTED_ARCHITECTURES`, `GGUF_TO_TRANSFORMERS_MAPPING`, and
`GGUF_TO_FAST_CONVERTERS`, but does **not** add it to transformers'
`CONFIG_MAPPING` (used by `AutoConfig.for_model`). When
`AutoTokenizer.from_pretrained` reads the GGUF file's metadata and encounters
`model_type: qwen35`, it calls `AutoConfig.for_model(model_type="qwen35")`
which raises `ValueError` because `qwen35` (no underscore) is absent from
`CONFIG_MAPPING`. Note: `qwen3_5` (with underscore) **is** present, indicating
a transformers version mismatch in how the architecture name is normalised.

This is a bug in the loader. Editing `third_party/tt_forge_models/` is out of
scope for this skill. The fix must be applied in the tt-forge-models repo:
`_patch_qwen35_support()` should also patch `CONFIG_MAPPING` to alias
`qwen35` → `Qwen3Config` (or whichever config class handles this arch).

## Measured (full model, defaults)
- Sample per second:  N/A (failed before compilation)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (no perf metrics generated)
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
- tests/benchmark/test_llms.py (added test_mradermacher_qwen3_5_4b_claude_4_6_opus_reasoning_distill_heretic_v3_i1_gguf)

## tt-forge-models submodule
no change
