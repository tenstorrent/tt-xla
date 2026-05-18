loader_path: third_party.tt_forge_models.chatts_8b.causal_lm.pytorch.loader
variant_id: 8B
arch: p150
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
failure_reason: "model forward bug: attention_mask.size(-1) called on NoneType in modeling_qwen3_ts.py:584 when cache_position is not None but no timeseries input is provided"

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

## Failure analysis
The test fails with `AttributeError: 'NoneType' object has no attribute 'size'`
in the model's custom `modeling_qwen3_ts.py` at line 584.

Root cause: `Qwen3TSForCausalLM.forward()` has an indentation bug in the
custom forward method. The block:

    if cache_position is not None:
        cache_position = torch.arange(
            attention_mask.size(-1) - inputs_embeds.size(1),
            attention_mask.size(-1),
            device=inputs_embeds.device
        )

…is at the same level as `if timeseries is not None and timeseries.shape[0] > 0:`,
so it runs in text-only mode (when `timeseries` is None). At that point
`attention_mask` has not been set by the time-series merge path, so it
remains `None` (our harness never passes it). Calling `.size(-1)` on
`None` raises `AttributeError`.

The fix belongs in the tt-forge-models loader or in the model's
`modeling_qwen3_ts.py` (filed as a bug upstream). Modifying model
definitions is out of scope for this skill.

## Measured (full model, defaults)
- Sample per second:  N/A (DONE_FAIL)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (not generated — test failed before device execution)
Achieved vs top_perf_samples_per_sec: N/A

### System
N/A

### Peak FLOPs
N/A

### Compute
N/A

### Inputs
N/A

### KV cache
N/A

### Params
N/A

### Roofline
N/A

## Files changed
- tests/benchmark/test_llms.py (test_chatts_8b function added)

## tt-forge-models submodule
no change
