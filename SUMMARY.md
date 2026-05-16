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
failure_reason: "GGUF architecture 'qwen35' not supported in transformers 5.2.0; supported qwen GGUF architectures are: qwen2, qwen2_moe, qwen3, qwen3_moe. Fix requires transformers update or loader change."

# Benchmark added: test_aaryank_qwen3_5_0_8b_gguf

## Test
tests/benchmark/test_llms.py::test_aaryank_qwen3_5_0_8b_gguf

## Model
- HF name:    AaryanK/Qwen3.5-0.8B-GGUF
- Loader:     third_party.tt_forge_models.aaryank_qwen3_5_0_8b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.QWEN3_5_0_8B_GGUF (= "0.8B_GGUF")

## Failure

**Error:** `ValueError: GGUF model with architecture qwen35 is not supported yet.`

Raised by `transformers==5.2.0` in `transformers/modeling_gguf_pytorch_utils.py:478`
at load time, before any TT device work begins.

The GGUF file `Qwen3.5-0.8B.q4_k_m.gguf` declares architecture `qwen35`.
`transformers` 5.2.0 supports only the following qwen GGUF variants:
`qwen2`, `qwen2_moe`, `qwen3`, `qwen3_moe`.

**Required fix (out of scope for this skill):**
- Either upgrade `transformers` to a version that adds `qwen35` GGUF support, OR
- Update the loader in `tt-forge-models` to load the weights without GGUF
  (e.g. convert to safetensors first, or use `from_pretrained` on the base HF repo).

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (model failed to load)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n300 (wormhole_b0, single-chip / n150 tier)

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (run did not complete)
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
- tests/benchmark/test_llms.py (added test_aaryank_qwen3_5_0_8b_gguf)
- SUMMARY.md

## tt-forge-models submodule
no change — submodule at c6c6fad0678fd039e96dbc30def5a6ef64fd80e8
