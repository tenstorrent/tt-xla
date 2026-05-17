loader_path: third_party.tt_forge_models.chatts_8b.causal_lm.pytorch.loader
variant_id: 8B
arch: n150
status: DONE_FAIL
test_function: test_chatts_8b
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: bfp_bf8
failure_reason: "model forward bug in modeling_qwen3_ts.py: cache_position recalculation block at line 584 accesses attention_mask.size(-1) when attention_mask=None (happens when timeseries=None and cache_position is provided); AttributeError: 'NoneType' object has no attribute 'size'"

# Benchmark added: test_chatts_8b

## Test
tests/benchmark/test_llms.py::test_chatts_8b

## Model
- HF name:    bytedance-research/ChatTS-8B
- Loader:     third_party.tt_forge_models.chatts_8b.causal_lm.pytorch.loader
- Variant:    ModelVariant.CHATTS_8B (8B)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150

## Failure detail

The test failed during CPU golden computation (prefill) before any TT hardware
involvement. The ChatTS-8B model uses a custom `modeling_qwen3_ts.py` module
that has a bug in `Qwen3TSForCausalLM.forward()`:

```python
# Line ~582-586 in modeling_qwen3_ts.py
if cache_position is not None:
    cache_position = torch.arange(
        attention_mask.size(-1) - inputs_embeds.size(1),  # <- FAILS
        attention_mask.size(-1),
        device=inputs_embeds.device
    )
```

This `cache_position` recalculation is only meaningful after
`_merge_input_ids_with_time_series_features()` expands the sequence (when
`timeseries` is provided). When `timeseries=None`, `attention_mask` is still
the value passed into `forward()` — which the benchmark harness leaves as
`None` (the standard practice for models that generate it internally).

The conditional should be nested inside `if timeseries is not None and
timeseries.shape[0] > 0:`, not at the outer level. This is a bug in the
model's custom HuggingFace module code and is not fixable from the test
or benchmarking infrastructure.

Full traceback:
```
tests/benchmark/benchmarks/llm_benchmark.py:406: in benchmark_llm_torch_xla
    cpu_prefill_logits, _ = generate_and_benchmark(
tests/benchmark/llm_utils/decode_utils.py:322: in generate_and_benchmark
    output = model(**input_args)
tests/benchmark/llm_utils/decode_utils.py:58: in forward
    output = self.model(
modeling_qwen3_ts.py:584: in forward
    attention_mask.size(-1) - inputs_embeds.size(1),
AttributeError: 'NoneType' object has no attribute 'size'
```

## Decode roofline (first decode graph, single-chip)
N/A — test failed before reaching TT hardware

## Files changed
- tests/benchmark/test_llms.py (added test_chatts_8b)
- .github/workflows/perf-bench-matrix.json (added chatts_8b entry)

## tt-forge-models submodule
no change
