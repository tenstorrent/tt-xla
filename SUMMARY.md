loader_path: third_party.tt_forge_models.ggml_org_qwen3_14b_gguf.causal_lm.pytorch.loader
variant_id: 14B_GGUF
arch: p150
status: DONE_PASS
test_function: test_ggml_org_qwen3_14b_gguf
samples_per_second: 3.352
ttft_ms: 1138.27
prefill_pcc: 0.998330
first_decode_pcc: 0.998108
top_perf_samples_per_sec: 23.1042
pct_of_target: 14.5
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Benchmark: test_ggml_org_qwen3_14b_gguf (p150)

## Model

- **Loader**: `third_party.tt_forge_models.ggml_org_qwen3_14b_gguf.causal_lm.pytorch.loader`
- **Variant**: `14B_GGUF`
- **HuggingFace**: `ggml-org/Qwen3-14B-GGUF` (Q4_K_M quantization, de-quantized to bfloat16)
- **Parameters**: 14.77B total, 13.99B effective (15.3 GB model memory)
- **Arch**: p150 (Blackhole, single chip)

## Infrastructure Fix

Fixed a bug in `tests/benchmark/benchmarks/llm_benchmark.py` where
`model_loader.get_weight_dtype_config_path()` was called unconditionally,
causing `AttributeError` for GGUF loaders (and any other loader that doesn't
implement this optional method). Changed `else:` to
`elif hasattr(model_loader, "get_weight_dtype_config_path"):`.

## Test Configuration

```python
optimization_level=2      # DEFAULT_OPTIMIZATION_LEVEL
trace_enabled=True        # default
experimental_weight_dtype="bfp_bf8"  # default
batch_size=32             # default
input_sequence_length=128 # default
required_pcc=0.94         # default
```

## Results (Full Model — All 48 Layers)

| Metric | Value |
|---|---|
| Sample per second | **3.352** |
| TTFT (ms) | **1138.27** |
| Prefill PCC | **0.998330** ✓ (required ≥ 0.94) |
| First decode PCC | **0.998108** ✓ (required ≥ 0.94) |
| Total test time | ~61 minutes |

## Roofline Analysis

| Metric | Value |
|---|---|
| Bound | **DRAM** |
| top_perf_samples_per_sec | 23.1042 |
| top_perf_time_ms | 43.28 ms |
| Measured samples/sec | 3.352 |
| % of roofline | **14.5%** |
| Measured decode step time | ~298 ms (vs 43.3 ms roofline) |

The model is DRAM-bound at 14.5% of the theoretical roofline (23.1 samples/sec).
Measured decode step time is ~298 ms vs the 43.3 ms DRAM-bandwidth limit, indicating
significant room for performance improvement (memory access patterns, op fusion, etc.).

## Bring-up Notes

- Bring-up test (num-layers=1, max-output-tokens=3): PASSED immediately after the
  `get_weight_dtype_config_path` infrastructure fix.
- Num-layers=1 full benchmark: ~84 samples/sec, 56.56ms TTFT.
- Full model (48 layers): 3.352 samples/sec, 1138ms TTFT.
- PCC excellent at both levels (0.998+), well above the 0.94 threshold.
- The GGUF model de-quantizes all 443 tensors from Q4_K_M to bfloat16 during loading
  (two separate XLA compilations: perf_model and logits_model, each with prefill+decode
  graphs — total 4 compilations, ~35-40 minutes of compile time).
