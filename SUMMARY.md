loader_path: third_party.tt_forge_models.dolphin3_llama3_1_bartowski_gguf.causal_lm.pytorch.loader
variant_id: 8B_Q4_K_M_GGUF
arch: p150
status: DONE_PASS
test_function: test_dolphin3_llama3_1_bartowski_gguf_8b_q4_k_m
samples_per_second: 31.849
ttft_ms: 311.2516
prefill_pcc: 0.998983
first_decode_pcc: 0.996274
top_perf_samples_per_sec: 42.5800
pct_of_target: 74.8
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Dolphin3 Llama3.1 8B Bartowski GGUF — Benchmark Results (p150)

## Model
- **Loader**: `third_party.tt_forge_models.dolphin3_llama3_1_bartowski_gguf.causal_lm.pytorch.loader`
- **Variant**: `8B_Q4_K_M_GGUF` (bartowski/Dolphin3.0-Llama3.1-8B-GGUF, Q4_K_M quantization)
- **Architecture**: p150 (blackhole, single chip)
- **HF Model**: `bartowski/Dolphin3.0-Llama3.1-8B-GGUF`

## Test Configuration
- **Test function**: `test_dolphin3_llama3_1_bartowski_gguf_8b_q4_k_m`
- **optimization_level**: 2 (DEFAULT_OPTIMIZATION_LEVEL)
- **trace_enabled**: True (default)
- **experimental_weight_dtype**: bfp_bf8 (default from test_llm)
- **batch_size**: 32 (default)
- **num_layers**: None (full model, 32 layers)

## Measured Performance
| Metric | Value |
|--------|-------|
| Samples per second | 31.85 |
| TTFT (ms) | 311.25 |
| Prefill PCC | 0.9990 ✓ (threshold: 0.94) |
| First decode PCC | 0.9963 ✓ (threshold: 0.94) |

## Roofline Analysis (DRAM-bound)
| Metric | Value |
|--------|-------|
| top_perf_samples_per_sec | 42.58 |
| top_perf_time_ms | 23.49 ms |
| Measured % of roofline | **74.8%** |
| dram_time_ms | 15.66 ms |

## Model Stats
- Parameters: 8.03B (effective: 7.50B)
- Model memory: 8.4 GB
- KV cache memory: 0.5 GB

## Notes
- GGUF model (already quantized Q4_K_M); `experimental_weight_dtype=bfp_bf8` is
  passed as a compiler option but per-tensor weight overrides are skipped because
  the GGUF loader does not implement `get_weight_dtype_config_path()`.
- Fix applied to `tests/benchmark/benchmarks/llm_benchmark.py`: added `hasattr()`
  guard before calling `model_loader.get_weight_dtype_config_path()` to support
  GGUF loaders that don't implement this method.
- Both prefill and decode PCC verified well above the 0.94 threshold.
