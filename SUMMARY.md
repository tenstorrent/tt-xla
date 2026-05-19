loader_path: third_party.tt_forge_models.chinese_alpaca_2.causal_lm.pytorch.loader
variant_id: 13B
arch: p150
status: DONE_FAIL
test_function: test_chinese_alpaca_2_13b
samples_per_second: 12.233788593080332
ttft_ms: 684.749036
prefill_pcc: 0.993442
first_decode_pcc: 0.752413
top_perf_samples_per_sec: 22.5947
pct_of_target: 54.2
roofline_bound: dram
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "First decode PCC consistently ~0.70-0.75 (required 0.94) across all tested configurations (bfp_bf8, no bfp_bf8, optimization_level 0-2); prefill PCC is fine (~0.99), suggesting KV cache numerical accumulation error in full 40-layer model decode step on p150"

# Benchmark added: test_chinese_alpaca_2_13b

## Test
tests/benchmark/test_llms.py::test_chinese_alpaca_2_13b

## Status: DONE_FAIL

## Model
- HF name:    hfl/chinese-alpaca-2-13b (LLaMA-2 13B fine-tuned for Chinese)
- Loader:     third_party.tt_forge_models.chinese_alpaca_2.causal_lm.pytorch.loader
- Variant:    ModelVariant.CHINESE_ALPACA_2_13B ("13B")
- Parameters: ~13.3B (KV cache: 3.125 GB)

## Failure Reason

First decode PCC consistently below the required 0.94 threshold regardless of optimization settings:

| Setting | Decode PCC |
|---------|-----------|
| opt_level=2, bfp_bf8 (default) | 0.752413 |
| opt_level=2, no weight dtype | 0.740157 |
| opt_level=1, bfp_bf8 | 0.699876 |
| opt_level=0, bfp_bf8 | ~0.70 |

The prefill PCC is excellent (0.993442) — issue is specifically in the decode step
with the full 40-layer model. The 1-layer model passes perfectly (decode PCC=0.999227),
confirming this is a numerical accumulation issue across transformer layers on p150.

## Test config landed
- optimization_level:        2 (DEFAULT_OPTIMIZATION_LEVEL)
- trace_enabled:             true (default)
- experimental_weight_dtype: bfp_bf8 (default)
- batch_size:                32

## Measured numbers (from PCC accuracy phase; perf benchmark not reached)
- Sample per second:         12.23 samples/sec (54.2% of roofline)
- TTFT (ms):                 684.75ms
- Prefill PCC:               0.993442 ✓
- First Decode PCC:          0.752413 ✗ (best; required: 0.94)

## Roofline (first decode graph)
- arch:                      blackhole (p150)
- bound:                     DRAM
- top_perf_samples_per_sec:  22.5947
- top_perf_time_ms:          44.26ms
- KV cache:                  3.125 GB (40 layers × 40 kv-heads × bfloat16)
- params effective:          12.84 GB

## Infrastructure fix included
tests/benchmark/benchmarks/llm_benchmark.py: Added `hasattr` guard around
`model_loader.get_weight_dtype_config_path()` to prevent AttributeError for
models without that method. Matches existing guard in dynamic_torch_model_tester.py.
