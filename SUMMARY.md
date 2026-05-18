loader_path: third_party.tt_forge_models.gemma_sea_guard_12b_2602_gguf.causal_lm.pytorch.loader
variant_id: SEA_Guard_12B_2602_GGUF
arch: p150
status: DONE_PASS
test_function: test_gemma_sea_guard_12b_2602_gguf
samples_per_second: 13.355470779250098
ttft_ms: 953.980684
prefill_pcc: 0.999988
first_decode_pcc: 0.999600
top_perf_samples_per_sec: 26.3289
pct_of_target: 50.7
roofline_bound: dram
optimization_level: 2
trace_enabled: null
experimental_weight_dtype: null
failure_reason: null

# Gemma SEA-Guard 12B 2602 GGUF — p150 Benchmark

## Model
- **Loader**: `third_party.tt_forge_models.gemma_sea_guard_12b_2602_gguf.causal_lm.pytorch.loader`
- **Variant**: `SEA_Guard_12B_2602_GGUF`
- **HF model**: `mradermacher/Gemma-SEA-Guard-12B-2602-GGUF` (Q4_K_M GGUF)
- **Params**: ~12.8B (effective ~11.8B; embeddings excluded)
- **Architecture**: Gemma3 with interleaved sliding/full attention (48 layers; every 6th is full attention)

## Hardware
- **Device**: p150 (blackhole, 1 chip, 110 worker cores, 512 GB/s DRAM)

## Bring-up Notes
- Model uses interleaved `sliding_attention` / `full_attention` layers (Gemma3-specific); the
  benchmark harness (`llm_benchmark.py` @ HEAD `a538fd74a`) already patches all layers to
  `full_attention` before compilation. Stale `.pyc` bytecode from a prior build had masked this
  fix; cleared with `find tests/benchmark -name "*.pyc" -delete` before running.
- Required `TT_MESH_GRAPH_DESC_PATH` pointing at the bundled p150 mesh graph descriptor (standard
  for blackhole single-chip runs).

## Test Configuration
- **Test function**: `test_gemma_sea_guard_12b_2602_gguf` (in `tests/benchmark/test_llms.py`)
- **optimization_level**: 2 (DEFAULT_OPTIMIZATION_LEVEL)
- **trace_enabled**: True (default)
- **experimental_weight_dtype**: bfp_bf8 (default)
- **batch_size**: 32
- **ISL**: 128

## Results (Full 48-Layer Model)
| Metric | Value |
|---|---|
| Status | **PASSED** |
| Prefill PCC | 0.999988 |
| First decode PCC | 0.999600 |
| Sample per second | **13.36** |
| TTFT (ms) | **953.98** |
| Wall clock | ~50 min (compilation + benchmark) |

## Roofline Analysis (Decode Graph)
| Metric | Value |
|---|---|
| Roofline bound | **DRAM** |
| top_perf_samples_per_sec | 26.33 |
| top_perf_time_ms | 37.98 |
| Achieved samples/s | 13.36 |
| **% of target** | **50.7%** |

The model is DRAM-bound at the roofline. Achieving ~50.7% of the theoretical ceiling is reasonable
for a 12B GGUF model with bfp_bf8 weights at batch_size=32 on a single p150 chip.
