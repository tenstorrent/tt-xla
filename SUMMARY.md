loader_path: third_party.tt_forge_models.sage_mm_qwen2_5_vl_7b_sft_rl_gguf.causal_lm.pytorch.loader
variant_id: 7B_SFT_RL_GGUF
arch: p150
status: DONE_PASS
test_function: test_sage_mm_qwen2_5_vl_7b_sft_rl_gguf
samples_per_second: 4.192496
ttft_ms: 1221.11
prefill_pcc: 0.992760
first_decode_pcc: 0.996773
top_perf_samples_per_sec: 46.0471
pct_of_target: 9.1
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: null

# SAGE-MM Qwen2.5-VL 7B SFT+RL GGUF — p150 Benchmark

## Model

- **HuggingFace**: `allenai/SAGE-MM-Qwen2.5-VL-7B-SFT_RL`
- **Loader**: `third_party.tt_forge_models.sage_mm_qwen2_5_vl_7b_sft_rl_gguf.causal_lm.pytorch.loader`
- **Variant**: `7B_SFT_RL_GGUF`
- **Architecture**: Qwen2-VL (VL model; language model extracted for causal LM benchmark)
- **Parameters**: ~7.6B total (7.07B effective after embedding exclusion)
- **Device**: p150 (blackhole, single-chip)

## Test Configuration

| Setting | Value |
|---|---|
| `optimization_level` | 2 (DEFAULT_OPTIMIZATION_LEVEL) |
| `trace_enabled` | True (DEFAULT_TRACE_ENABLED) |
| `experimental_weight_dtype` | `bfp_bf8` |
| `batch_size` | 32 |
| `input_sequence_length` | 128 |
| `data_format` | bfloat16 |

## Results

| Metric | Value |
|---|---|
| Prefill PCC | **0.992760** ✓ (threshold: 0.94) |
| First decode PCC | **0.996773** ✓ (threshold: 0.94) |
| Sample per second | **4.192** |
| TTFT (ms) | **1221.1** |
| Total test time | ~316 s (5:16) |

## Roofline Analysis

| Metric | Value |
|---|---|
| Roofline bound | DRAM |
| Target (top_perf_samples_per_sec) | 46.05 |
| Measured | 4.192 |
| % of target | **9.1%** |
| DRAM time/step (ideal) | 14.48 ms |
| Measured decode step | ~239 ms |

The model is DRAM-bound. The 9.1% efficiency at batch_size=32 reflects typical single-chip
utilization for a 7B model with bfp_bf8 weights on p150. Each decode step measured ~239 ms
vs 21.7 ms theoretical minimum.

## Notes

This model loads `Qwen2_5_VLForConditionalGeneration` and extracts the language model
(`Qwen2ForCausalLM`) internally, discarding the visual encoder. The full VL weight set
(729 parameters) is downloaded but only the LM weights are used for this benchmark.

## Infrastructure Fix

`tests/benchmark/benchmarks/llm_benchmark.py` was patched to guard the
`get_weight_dtype_config_path()` call with a defensive `getattr`, so loaders
that don't implement this optional method no longer crash:

```python
# Before:
weight_dtype_config = model_loader.get_weight_dtype_config_path()

# After:
_get_fn = getattr(model_loader, "get_weight_dtype_config_path", None)
weight_dtype_config = _get_fn() if _get_fn is not None else None
```

## Files Changed

- `tests/benchmark/test_llms.py` — added `test_sage_mm_qwen2_5_vl_7b_sft_rl_gguf`
- `tests/benchmark/benchmarks/llm_benchmark.py` — defensive `getattr` for `get_weight_dtype_config_path`
