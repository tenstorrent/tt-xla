loader_path: third_party.tt_forge_models.code_qwen_1_5_gguf.causal_lm.pytorch.loader
variant_id: 7B_Chat_GGUF
arch: p150
status: DONE_PASS
test_function: test_code_qwen_1_5_7b_chat_gguf
samples_per_second: 4.643783608121185
ttft_ms: 779.551206
prefill_pcc: 0.997916
first_decode_pcc: 0.998214
top_perf_samples_per_sec: 47.2440
pct_of_target: 9.8
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# CodeQwen 1.5 7B Chat GGUF — Benchmark Results (p150)

## Model
- **Loader**: `third_party.tt_forge_models.code_qwen_1_5_gguf.causal_lm.pytorch.loader`
- **Variant**: `7B_Chat_GGUF`
- **HuggingFace**: `Qwen/CodeQwen1.5-7B-Chat-GGUF` (codeqwen-1_5-7b-chat-q4_k_m.gguf)
- **Test function**: `tests/benchmark/test_llms.py::test_code_qwen_1_5_7b_chat_gguf`
- **Hardware**: p150 (Blackhole, 1 chip)

## Measured Performance (full 32-layer model)

| Metric               | Value          |
|----------------------|----------------|
| Samples/sec          | 4.64           |
| TTFT (ms)            | 779.6          |
| Prefill PCC          | 0.9979 ✅       |
| First decode PCC     | 0.9982 ✅       |

## Roofline Analysis (first decode graph)

| Metric                    | Value          |
|---------------------------|----------------|
| Bound                     | DRAM           |
| top_perf_samples_per_sec  | 47.24          |
| top_perf_time_ms          | 21.17 ms       |
| % of target               | 9.8%           |

The 9.8% of roofline is expected for a Q4_K_M GGUF quantized model, which requires
dequantization overhead at inference time not captured in the roofline model.

## Final Test Configuration

- `optimization_level=2` (DEFAULT_OPTIMIZATION_LEVEL)
- `trace_enabled=True` (default)
- `experimental_weight_dtype="bfp_bf8"` (DEFAULT_EXPERIMENTAL_WEIGHT_DTYPE)
- `batch_size=32` (default)

## Infrastructure Fix

Applied fix to `tests/benchmark/benchmarks/llm_benchmark.py`: changed unconditional
`else:` → `elif hasattr(model_loader, "get_weight_dtype_config_path"):` so loaders
that do not implement the optional `get_weight_dtype_config_path()` method work
correctly with the benchmark harness.
