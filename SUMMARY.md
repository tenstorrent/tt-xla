loader_path: third_party.tt_forge_models.claude2_alpaca_13b_gguf.causal_lm.pytorch.loader
variant_id: 13B_GGUF
arch: p150
status: DONE_FAIL
test_function: test_claude2_alpaca_13b_gguf
samples_per_second: 12.826982366001024
ttft_ms: 390.3101
prefill_pcc: 0.997294
first_decode_pcc: 0.467600
top_perf_samples_per_sec: 22.7801
pct_of_target: 56.3
roofline_bound: dram
optimization_level: 1
trace_enabled: false
experimental_weight_dtype: bfp_bf8
failure_reason: "First decode PCC failed across all optimization levels: opt=2 Fatal Python Bus error during warmup, opt=1 decode PCC=0.4676 (required 0.94), opt=0 decode PCC denominator=zero error"

# Benchmark: test_claude2_alpaca_13b_gguf (p150)

## Model
- **Loader**: `third_party.tt_forge_models.claude2_alpaca_13b_gguf.causal_lm.pytorch.loader`
- **Variant**: `13B_GGUF`
- **HF Model**: TheBloke/claude2-alpaca-13B-GGUF (Q4_K_M quantization)
- **Architecture**: LLaMA-2 13B (40 layers)
- **Test function**: `test_claude2_alpaca_13b_gguf`

## Result: DONE_FAIL

The model consistently fails first decode PCC across all optimization levels on p150.
Prefill PCC passes at all tested levels, but the first decode step produces wrong results.

## Performance (optimization_level=1, trace_enabled=False)
- **Samples per second**: 12.83
- **TTFT (ms)**: 390.31
- **Roofline (top_perf_samples_per_sec)**: 22.78
- **% of roofline target**: 56.3%
- **Roofline bound**: DRAM
- **Prefill PCC**: 0.9973 ✓ (required 0.94)
- **First decode PCC**: 0.4676 ✗ (required 0.94)

## Optimization Level Sweep (all with trace_enabled=False, bfp_bf8)

| optimization_level | Result                              |
|--------------------|-------------------------------------|
| 0                  | FAIL: decode PCC denominator=0 error |
| 1                  | FAIL: decode PCC=0.4676 < 0.94      |
| 2                  | CRASH: Fatal Python Bus error during warmup |

## 1-layer Sanity Check
- `--num-layers 1`, opt=2, trace=False: **PASSED** (Prefill PCC=0.9990, Decode PCC=0.9620)
- The 1-layer test passes, confirming model loads and compiles correctly
- Full 40-layer model fails decode PCC, suggesting KV-cache or memory issue in decode path

## Hardware
- Device: Blackhole p300c (single chip, p150 arch)
- Chip count: 1
- DRAM bandwidth: 512 GB/s

## Configuration (hard-coded in test)
```python
optimization_level=1
trace_enabled=False  # Trace disabled: Bus error during warmup with trace=True on full 40-layer model
experimental_weight_dtype=bfp_bf8  # default
```

## Roofline Analysis (from extract_perf_targets.py)
- Total parameters: 13.0B (effective: 12.9B)
- KV cache memory: 3.125 GB
- Model weights memory (bf16): 24.24 GB
- Decode bound: DRAM (dram_time=29.27ms, top_perf=22.78 S/s)
