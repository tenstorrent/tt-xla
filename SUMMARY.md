loader_path: third_party.tt_forge_models.dream_omni_2_gguf.causal_lm.pytorch.loader
variant_id: Q4_K_M_GGUF
arch: p150
status: DONE_PASS
test_function: test_dream_omni_2_gguf
samples_per_second: 4.175951996216374
ttft_ms: 1241.282881
prefill_pcc: 0.992143
first_decode_pcc: 0.995485
top_perf_samples_per_sec: 46.0471
pct_of_target: 9.1
roofline_bound: dram
optimization_level: 0
trace_enabled: true
experimental_weight_dtype: null
failure_reason: null

# DreamOmni2 GGUF Benchmark — p150

## Model
- **Loader**: `third_party.tt_forge_models.dream_omni_2_gguf.causal_lm.pytorch.loader`
- **Variant**: `Q4_K_M_GGUF`
- **Model**: DreamOmni2 7.6B (`xiabs/DreamOmni2`, subfolder `vlm-model`)
- **Architecture**: Qwen2_5_VLForConditionalGeneration (Vision-Language Model loaded as causal LM)
- **Hardware**: Blackhole p150 (p300c)

## Test Configuration
- **Test function**: `test_dream_omni_2_gguf` in `tests/benchmark/test_llms.py`
- **Harness**: `test_llm` (single-chip)
- **`optimization_level`**: 0 (hard-coded — opt=2 hangs on p150 with this VLM model)
- **`trace_enabled`**: True (default)
- **`experimental_weight_dtype`**: not set
- **`batch_size`**: 32 (default)

## Measured Performance (full model, p150)
| Metric | Value |
|---|---|
| Samples/sec (decode) | 4.176 |
| TTFT (ms) | 1241.3 |
| Prefill PCC | 0.992143 pass |
| First decode PCC | 0.995485 pass |
| % of roofline target | 9.1% |

## Roofline Analysis (decode graph)
- **Bound**: DRAM
- **Top perf (samples/sec)**: 46.0471
- **Top perf time (ms)**: 21.7169
- **Effective param memory**: ~7.0 GB (7B params effective)
- **KV cache memory**: 0.22 GB

The 9.1% of roofline is expected at optimization_level=0 which keeps all tensors in DRAM
without SRAM push optimizations. Attempts with opt=2 caused a hardware hang (>43 min) on p150
with this VLM architecture.

## Infrastructure Fix
Added guard in tests/benchmark/benchmarks/llm_benchmark.py so that the harness uses
getattr(model_loader, "get_weight_dtype_config_path", lambda: None)() rather than calling
the method unconditionally -- required because the DreamOmni2 loader does not implement
get_weight_dtype_config_path. This is a general correctness fix for any loader that omits it.

## Submodule Note
The working loader (at third_party/tt_forge_models commit 30c94449f5) loads from
xiabs/DreamOmni2 (safetensors). The current HEAD of tt-forge-models uses the GGUF variant
(rafacost/DreamOmni2-7.6B-GGUF) with an incorrect filename. This benchmark pins the
submodule to 30c94449f5 where the working loader resides.
