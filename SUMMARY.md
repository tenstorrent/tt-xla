loader_path: third_party.tt_forge_models.anomamind_8b_i1_gguf.causal_lm.pytorch.loader
variant_id: 8B_I1_GGUF
arch: p150
status: DONE_PASS
test_function: test_anomamind_8b_i1_gguf
samples_per_second: 3.983186045863132
ttft_ms: 2813.622446
prefill_pcc: 0.998560
first_decode_pcc: 0.998799
top_perf_samples_per_sec: 42.0551
pct_of_target: 9.5
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# AnomaMind 8B i1 GGUF — Benchmark Summary

## Model
- **Loader**: `third_party.tt_forge_models.anomamind_8b_i1_gguf.causal_lm.pytorch.loader`
- **Variant**: `8B_I1_GGUF`
- **HuggingFace model**: `mradermacher/AnomaMind-8B-i1-GGUF` (GGUF file: `AnomaMind-8B.i1-Q4_K_M.gguf`)
- **Parameters**: ~8.19B (~7.57B effective, embedding excluded)
- **Model size**: ~8.65 GB

## Hardware
- **Architecture**: p150 (Blackhole)
- **Chip count**: 1 (single chip)

## Test Config
- `optimization_level`: 2
- `trace_enabled`: true
- `experimental_weight_dtype`: bfp_bf8
- `batch_size`: 32
- `isl`: 128

## Bring-up Notes

### Dependencies fixed
1. **Missing `gguf>=0.10.0`**: Installed `gguf-0.19.0` — required by HuggingFace transformers to load GGUF-format weights.
2. **`get_weight_dtype_config_path` missing**: Fixed general infra bug in `llm_benchmark.py` — changed `else:` to `elif hasattr(model_loader, "get_weight_dtype_config_path"):` to avoid `AttributeError` on loaders that don't implement this optional method.

### Compilation timeline (full model, batch_size=32, isl=128)
| Graph | Description             | Binary written | Duration |
|-------|-------------------------|----------------|----------|
| g0    | Prefill (no logits)     | 11:04 UTC      | ~10 min  |
| g1    | Decode (no logits)      | 11:50 UTC      | ~46 min  |
| g2    | Prefill (with logits)   | 12:01 UTC      | ~11 min  |
| g3    | Decode (with logits)    | ~12:12 UTC     | ~11 min  |

Total end-to-end test time: ~100 minutes (mostly cold kernel compilation for first run).

## Results

| Metric             | Value              |
|--------------------|--------------------|
| Samples/sec        | **3.983**          |
| TTFT (ms)          | 2813.6             |
| Prefill PCC        | 0.998560 ✓         |
| First decode PCC   | 0.998799 ✓         |
| Status             | **PASS**           |

## Roofline Analysis (first decode graph)

| Metric                  | Value              |
|-------------------------|--------------------|
| Bound                   | DRAM               |
| top_perf_samples_per_sec | 42.055 sps        |
| top_perf_time_ms        | 23.778 ms          |
| Measured samples/sec    | 3.983 sps          |
| % of roofline target    | **9.5%**           |

The 9.5% roofline efficiency is expected at this stage — the full DRAM bandwidth ceiling requires further runtime optimizations (pipelining, op fusion). Both PCC verifications passed well above threshold.
