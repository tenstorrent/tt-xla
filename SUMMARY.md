loader_path: third_party.tt_forge_models.dolphin_2_9_3_mistral_nemo_12b_gguf.causal_lm.pytorch.loader
variant_id: Q4_K_M
arch: p150
status: DONE_PASS
test_function: test_dolphin_2_9_3_mistral_nemo_12b_gguf
samples_per_second: 3.52848486246632
ttft_ms: 1039.240677
prefill_pcc: 0.997271
first_decode_pcc: 0.997413
top_perf_samples_per_sec: 27.7857
pct_of_target: 12.7
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# Dolphin 2.9.3 Mistral Nemo 12B GGUF (Q4_K_M) — p150 Benchmark

## Model
- **Loader**: `third_party.tt_forge_models.dolphin_2_9_3_mistral_nemo_12b_gguf.causal_lm.pytorch.loader`
- **Variant**: `Q4_K_M`
- **HuggingFace**: `dphn/dolphin-2.9.3-mistral-nemo-12b-gguf`
- **Parameters**: ~12.2B (effective: ~11.6B)
- **Architecture**: Mistral Nemo 12B, 40 layers

## Hardware
- **Arch**: p150 (Blackhole)
- **Chip count**: 1 (single device)

## Test Configuration
- **Test function**: `test_dolphin_2_9_3_mistral_nemo_12b_gguf`
- **Batch size**: 32
- **Optimization level**: 2
- **Trace enabled**: True
- **Weight dtype**: bfp_bf8
- **Input sequence length**: 128

## Results (Full Model)
| Metric | Value |
|--------|-------|
| Sample per second | 3.528 |
| TTFT (ms) | 1039.24 |
| Prefill PCC | 0.997271 ✅ |
| First decode PCC | 0.997413 ✅ |
| Status | PASSED |

## Roofline Analysis (Decode Graph)
| Metric | Value |
|--------|-------|
| Bound | DRAM |
| Top perf (samples/sec) | 27.7857 |
| Top perf time (ms) | 35.9898 |
| DRAM time (ms) | 23.9932 |
| Achieved (% of target) | 12.7% |

## Notes
- Bring-up (1-layer test) passed: prefill PCC=0.999478, decode PCC=0.999499, ~87.6 samples/sec
- Full model test passed on first attempt with default settings (optimization_level=2, trace=True, bfp_bf8)
- General infrastructure fix applied to `tests/benchmark/benchmarks/llm_benchmark.py`: changed unconditional `get_weight_dtype_config_path()` call to use `hasattr` guard (fixes AttributeError for loaders that don't implement this optional method)
- Achieved throughput is 12.7% of DRAM-bound roofline (3.53 vs 27.79 samples/sec) — consistent with other models at this stage
