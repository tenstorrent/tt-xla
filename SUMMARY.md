loader_path: third_party.tt_forge_models.aaryank_qwen3_5_0_8b_gguf.causal_lm.pytorch.loader
variant_id: QWEN3_5_0_8B_GGUF
arch: n150
status: DONE_FAIL
test_function: test_aaryank_qwen3_5_0_8b_gguf
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
failure_reason: "aaryank_qwen3_5_0_8b_gguf loader missing qwen35 GGUF architecture patch at submodule HEAD c6c6fad06; transformers 5.2.0 raises ValueError: GGUF model with architecture qwen35 is not supported yet. Fix: apply _patched_load_gguf_checkpoint (qwen35→qwen3 remap) to this loader as was done for aaryank_qwen3_5_9b_gguf in commit 6282ecad2e on a different branch."

# Benchmark added: test_aaryank_qwen3_5_0_8b_gguf

## Test
tests/benchmark/test_llms.py::test_aaryank_qwen3_5_0_8b_gguf

## Model
- HF name:    AaryanK/Qwen3.5-0.8B-GGUF
- Loader:     third_party.tt_forge_models.aaryank_qwen3_5_0_8b_gguf.causal_lm.pytorch.loader
- Variant:    QWEN3_5_0_8B_GGUF ("0.8B_GGUF")

## Failure
The loader calls `AutoTokenizer.from_pretrained(..., gguf_file="Qwen3.5-0.8B.q4_k_m.gguf")`.
`transformers` 5.2.0 parses the GGUF file's `general.architecture` field as `qwen35`, which is
**not** in `GGUF_SUPPORTED_ARCHITECTURES`. Transformers 5.2.0 knows: `qwen2`, `qwen2_moe`, `qwen3`, `qwen3_moe`.

Error (from `transformers/modeling_gguf_pytorch_utils.py:478`):
```
ValueError: GGUF model with architecture qwen35 is not supported yet.
```

The fix pattern (used for `aaryank_qwen3_5_9b_gguf` in commit `6282ecad2e`) is to add a
`_patched_load_gguf_checkpoint` that registers `qwen35` in `GGUF_SUPPORTED_ARCHITECTURES`
and remaps it to `qwen3`. Those commits exist on a separate branch, not in the current
submodule HEAD (`c6c6fad06` = tip of `arch-c-36-tt-xla-dev/nsmith/2026-04-22_16-58/hf-bringup-25`).

Editing `third_party/tt_forge_models/` is out of scope for this skill.

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (model failed to load)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n300 (n150 single-chip, Wormhole)

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not reach compilation)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        n150
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
- tests/benchmark/test_llms.py (test stub added)
- SUMMARY.md

## tt-forge-models submodule
no change
