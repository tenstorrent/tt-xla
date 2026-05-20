loader_path: third_party.tt_forge_models.ggml_org_qwen3_14b_gguf.causal_lm.pytorch.loader
variant_id: 14B_GGUF
arch: n150
status: DONE_FAIL
test_function: test_ggml_org_qwen3_14b_gguf
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: null
trace_enabled: null
experimental_weight_dtype: null
failure_reason: "model size ~14B exceeds 10B single-chip capacity on n150"

# Benchmark added: ggml_org_qwen3_14b_gguf

## Test
tests/benchmark/test_llms.py::test_ggml_org_qwen3_14b_gguf

## Model
- HF name:    ggml-org/Qwen3-14B-GGUF
- Loader:     third_party.tt_forge_models.ggml_org_qwen3_14b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GGML_ORG_QWEN3_14B_GGUF ("14B_GGUF")

## Early Exit Reason
Model size ~14B exceeds the 10B single-chip capacity on n150 (wormhole_b0). Weight transfer
will OOM before inference can begin. This model requires p150 (blackhole, ~25B cap) or a
multi-chip (TP) configuration. No test code was added; no weights were downloaded.

## Test config landed
- optimization_level:        N/A (early exit)
- trace_enabled:             N/A (early exit)
- experimental_weight_dtype: N/A (early exit)
- batch_size:                N/A (early exit)
- input_sequence_length:     N/A (early exit)
- required_pcc:              N/A (early exit)

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (early exit — no pytest run performed)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch:                        wormhole_b0
- chip_count_in_system_desc:   1
- single_chip_assumption:      true
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
- SUMMARY.md (this file)

## tt-forge-models submodule
no change
