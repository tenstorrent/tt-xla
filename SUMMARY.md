loader_path: third_party.tt_forge_models.029_shisa_gamma_7b_v1_v2new_dpo405b_i1_gguf.causal_lm.pytorch.loader
variant_id: 029_SHISA_GAMMA_7B_V1_V2NEW_DPO405B_I1_Q4_K_M_GGUF
arch: p150
status: DONE_FAIL
test_function: test_029_shisa_gamma_7b_v1_v2new_dpo405b_i1_q4_k_m_gguf
samples_per_second: 3.165
ttft_ms: 938.4
prefill_pcc: 0.9993
first_decode_pcc: 0.5880
top_perf_samples_per_sec: 44.855
pct_of_target: 7.1
roofline_bound: dram
optimization_level: 1
trace_enabled: true
experimental_weight_dtype: ""
failure_reason: "First decode PCC consistently fails at ~0.588 (required 0.94) on full 32-layer model across all configurations tested: opt_level=1+bfp_bf8 (PCC=0.565), opt_level=2+bfp_bf8 (PCC=0.673), opt_level=2+no_bfp_bf8 (PCC=0.587), opt_level=1+no_bfp_bf8 (PCC=0.588); prefill PCC passes (0.999); 1-layer decode passes (PCC=0.9997); root cause likely Q4_K_M GGUF quantization interaction with TT-XLA multi-layer decode path; also required llm_benchmark.py fix (elif hasattr guard) for get_weight_dtype_config_path absent on GGUF loaders"

# 029-Shisa-Gamma-7B-v1 v2new DPO405B i1 Q4_K_M GGUF Benchmark — p150

## Model

- **Loader**: `third_party.tt_forge_models.029_shisa_gamma_7b_v1_v2new_dpo405b_i1_gguf.causal_lm.pytorch.loader`
- **Variant**: `029_SHISA_GAMMA_7B_V1_V2NEW_DPO405B_I1_Q4_K_M_GGUF`
- **Architecture**: p150 (Tenstorrent Blackhole)
- **Parameters**: ~7.2B (Q4_K_M GGUF quantized, effective 13.2GB)

## Result: DONE_FAIL

### PCC Analysis (Full Model, 32 layers)

| Configuration | Prefill PCC | Decode PCC | Status |
|---|---|---|---|
| opt_level=1 + bfp_bf8 | 0.999 | 0.565 | ❌ FAIL |
| opt_level=2 + bfp_bf8 | 0.999 | 0.673 | ❌ FAIL |
| opt_level=2 + no bfp_bf8 | 0.999 | 0.587 | ❌ FAIL |
| opt_level=1 + no bfp_bf8 | 0.9993 | 0.5880 | ❌ FAIL |

### 1-Layer Validation

- Prefill PCC: 0.9997
- First decode PCC: 0.9998
- ✅ PASSES — confirms per-layer computation is correct

### Root Cause Analysis

The decode PCC consistently fails at 0.565–0.673 across ALL configuration combinations. Key observations:

1. **Prefill always passes** (~0.999): forward pass through all 32 layers is numerically accurate.
2. **1-layer decode passes** (0.9997): decode computation for a single layer is correct.
3. **Full 32-layer decode fails** (~0.588): error accumulates across 32 decode steps.
4. **Configuration-independent**: neither opt_level nor bfp_bf8 makes a meaningful difference.

This pattern is consistent with a numerical accuracy issue in the multi-layer KV-cache decode path on TT-XLA for Q4_K_M GGUF models. The ~0.588 decode PCC is well below the 0.94 requirement.

## Performance Numbers (at failed PCC)

| Metric | Value |
|---|---|
| Sample per second | 3.165 |
| TTFT | 938.4 ms |
| Prefill PCC | 0.9993 |
| First Decode PCC | 0.5880 |

## Roofline Analysis

| Metric | Value |
|---|---|
| Top perf (samples/sec) | 44.855 |
| Current (samples/sec) | 3.165 |
| % of roofline | 7.1% |
| Roofline bound | DRAM |

## Infrastructure Fix Required

The `tests/benchmark/benchmarks/llm_benchmark.py` required a fix for GGUF loaders:
- **Issue**: `else: weight_dtype_config = model_loader.get_weight_dtype_config_path()` crashes for GGUF loaders that don't implement this method.
- **Fix**: Changed `else:` to `elif hasattr(model_loader, "get_weight_dtype_config_path"):` to guard the call.

## Final Test Config

```python
optimization_level=1,     # DRAM-only (opt=2 SRAM doesn't help PCC)
experimental_weight_dtype="",  # disabled — bfp_bf8 on Q4_K_M causes double quantization
trace_enabled=True,        # default
batch_size=32,             # default
```
