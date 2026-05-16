loader_path: third_party.tt_forge_models.agentflow_slime_agentic_qwen2_5_gguf.causal_lm.pytorch.loader
variant_id: 7B_GGUF
arch: n150
status: DONE_FAIL
test_function: test_agentflow_slime_agentic_qwen2_5_gguf
samples_per_second: 19.18
ttft_ms: 613.76
prefill_pcc: 0.975119
first_decode_pcc: 0.930374
top_perf_samples_per_sec: 25.9015
pct_of_target: 74.0
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: null
failure_reason: "full model decode PCC failed: 0.930374 < 0.94 required with bfp_bf8; bfp_bf8 removed in final test commit (experimental_weight_dtype='') but this fix could not be verified — device became unresponsive after full model run and requires server reboot"

# Benchmark added: test_agentflow_slime_agentic_qwen2_5_gguf

## Test
tests/benchmark/test_llms.py::test_agentflow_slime_agentic_qwen2_5_gguf

## Model
- HF name:    mradermacher/AgentFlow_Slime_Agentic_Qwen2.5_7B-i1-GGUF
- Loader:     third_party.tt_forge_models.agentflow_slime_agentic_qwen2_5_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.AGENTFLOW_SLIME_AGENTIC_QWEN_2_5_7B_GGUF (= "7B_GGUF")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: none (bfp_bf8 disabled due to decode PCC regression)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Bring-up notes
- `gguf>=0.10.0` package not in venv by default; installed manually. Added to pyreq in perf-bench-matrix.json.
- `/home/ogi` partition was full (100%); model downloaded to `/tmp/hf_cache` via `HF_HOME=/tmp/hf_cache`.
- `get_weight_dtype_config_path()` missing from loader caused `AttributeError` in `llm_benchmark.py`. Fixed by adding `hasattr` guard (general fix, not model-specific).
- `--num-layers 1` with bfp_bf8: PASSED (prefill PCC=0.976, decode PCC=0.987).
- Full model with bfp_bf8: FAILED decode PCC (0.930374 < 0.94). Performance phase completed successfully.
- After PCC failure, device became unresponsive (Ethernet core timeout). Further testing (without bfp_bf8) was blocked.
- Test committed with `experimental_weight_dtype=""` (bfp_bf8 disabled) as the likely fix, but this has not been verified on the full model.

## Measured (full model, bfp_bf8 — performance phase only)
- Sample per second:  19.18
- TTFT (ms):          613.76
- Prefill PCC:        0.975119 (PASS)
- First decode PCC:   0.930374 (FAIL — required 0.94)
- Wall clock:         0:15:48
- Hardware:           n150 (wormhole_b0, 2-chip system used as single-chip)

## Decode roofline (first decode graph, single-chip)
Source JSON: tt_xla_agentflow_slime_agentic_qwen2_5_gguf_perf_metrics_1.json
Achieved vs top_perf_samples_per_sec: 74.0% (19.18 / 25.9015)

### System
- arch:                        wormhole_b0
- chip_count_in_system_desc:   2
- single_chip_assumption:      True
- worker_grid_cores:           64
- dram_bandwidth_bytes_per_sec: 288000000000

### Peak FLOPs
- lofi:  256000000000000
- hifi2: 128000000000000
- hifi3: 85333333333333
- hifi4: 64000000000000

### Compute
- total_flops:             452502421632
- breakdown.matmul:        422903283840
- breakdown.linear:        29599137792
- breakdown.conv2d:        0
- breakdown.sparse_matmul: 0

### Inputs
- count:        33
- memory_bytes: 132

### KV cache
- count:        117440512
- memory_bytes: 234881024
- memory_gb:    0.21875

### Params
- count:                  7615616707
- effective_count:        7070619331
- memory_bytes:           8602840840
- memory_gb:              8.012
- effective_memory_bytes: 7512846088
- effective_memory_gb:    6.997
- embedding_count:        544997376
- embedding_memory_bytes: 1089994752

### Roofline
- bound:                    dram
- top_perf_samples_per_sec: 25.9015
- top_perf_time_ms:         38.6077
- dram_time_ms:             25.7385
- compute_time_ms_lofi:     1.7676
- compute_time_ms_hifi2:    3.5352
- compute_time_ms_hifi3:    5.3028
- compute_time_ms_hifi4:    7.0704

## Files changed
- tests/benchmark/test_llms.py (new test function added)
- tests/benchmark/benchmarks/llm_benchmark.py (hasattr guard for get_weight_dtype_config_path)
- .github/workflows/perf-bench-matrix.json (new entry with gguf>=0.10.0 in pyreq)

## tt-forge-models submodule
no change
