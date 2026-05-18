loader_path: third_party.tt_forge_models.andy_4_1_i1_gguf.causal_lm.pytorch.loader
variant_id: 4_1_I1_GGUF
arch: p150
status: DONE_FAIL
test_function: test_andy_4_1_i1_gguf
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
failure_reason: "loader bug: GGUF file (Andy-4.1.i1-Q4_K_M.gguf) architecture is 2B (hidden_size=2048, num_kv_heads=8) but PROCESSOR_BASE=Qwen/Qwen3-VL-4B-Instruct config is 4B (hidden_size=4096, num_attention_heads=32); ignore_mismatched_sizes=True causes silent weight reinit; RuntimeError: index_copy_(): Source/destination tensor must have same slice shapes (32,8,128 vs 32,32,128) in KV cache during CPU prefill"

# Benchmark added: test_andy_4_1_i1_gguf

## Test
tests/benchmark/test_llms.py::test_andy_4_1_i1_gguf

## Model
- HF name:    mradermacher/Andy-4.1-i1-GGUF
- Loader:     third_party.tt_forge_models.andy_4_1_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.ANDY_4_1_I1_GGUF (value: "4_1_I1_GGUF")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Failure
The test fails during CPU prefill (golden reference generation) with:

```
RuntimeError: index_copy_(): Source/destination tensor must have same slice shapes.
Destination slice shape: 32 8 128 at dimension 2
Source slice shape: 32 32 128 at dimension 0.
```

This is a loader-level issue: the GGUF file `Andy-4.1.i1-Q4_K_M.gguf` contains a 2B model
(num_kv_heads=8, hidden_size=2048) but `PROCESSOR_BASE = "Qwen/Qwen3-VL-4B-Instruct"` causes
transformers to build a 4B Qwen3VL architecture (num_attention_heads=32, hidden_size=4096).
The `ignore_mismatched_sizes=True` flag silently reinitializes the mismatched weights, leaving
a structurally inconsistent model that crashes during the KV cache update.

The fix belongs in `third_party/tt_forge_models/andy_4_1_i1_gguf/causal_lm/pytorch/loader.py`
— either the PROCESSOR_BASE should be a 2B VL model (if Andy-4.1 is 2B), or the GGUF file
should be replaced with a 4B variant. No changes to `third_party/tt_forge_models/` are made
here per skill rules.

## General harness fixes included (llm_benchmark.py)
Two general infrastructure fixes were applied to support VL/multimodal models:

1. **Processor tokenizer support**: The harness previously did `tokenizer = model_loader.tokenizer`
   unconditionally. Changed to check for `model_loader.tokenizer` first, fall back to
   `model_loader.processor.tokenizer` (for VL loaders that expose a processor), raise
   `AttributeError` if neither is available.

2. **VL model config fix**: The harness used `model.config` for KV cache construction.
   Changed to use `model.language_model.config` when present (for Qwen3VL and similar composite
   VL architectures where KV-relevant config lives on the inner language_model, not the wrapper).

Both fixes are backward-compatible and general; they do not affect existing standard LLM tests.

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150 (Blackhole)

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A — test did not reach device execution
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        p150 / Blackhole
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
- tests/benchmark/test_llms.py (added test_andy_4_1_i1_gguf function)
- tests/benchmark/benchmarks/llm_benchmark.py (general VL model harness fixes)
- .github/workflows/perf-bench-matrix.json (added andy_4_1_i1_gguf entry)
- SUMMARY.md (this file)

## tt-forge-models submodule
no change — submodule at 46b08cd10d
