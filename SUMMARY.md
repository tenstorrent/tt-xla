loader_path: third_party.tt_forge_models.egan_ai_qwen3_5_9b_terminal_merge.causal_lm.pytorch.loader
variant_id: 9B_Terminal_Merge
arch: p150
status: DONE_FAIL
test_function: test_egan_ai_qwen3_5_9b_terminal_merge
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
failure_reason: "AttributeError: 'StaticCache' object has no attribute 'has_previous_state' in transformers/models/qwen3_5/modeling_qwen3_5.py:525 — Qwen3.5 hybrid model expects Qwen3_5DynamicCache but benchmark harness passes StaticCache (transformers==5.2.0 incompatibility)"

# Benchmark added: test_egan_ai_qwen3_5_9b_terminal_merge

## Test
tests/benchmark/test_llms.py::test_egan_ai_qwen3_5_9b_terminal_merge

## Model
- HF name:    EganAI/qwen3.5-9b-terminal-merge
- Loader:     third_party.tt_forge_models.egan_ai_qwen3_5_9b_terminal_merge.causal_lm.pytorch.loader
- Variant:    ModelVariant.QWEN3_5_9B_TERMINAL_MERGE

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
- Hardware:           p150

## Failure Details
The test failed during the `--num-layers 1 --max-output-tokens 3` bring-up run with:

```
AttributeError: 'StaticCache' object has no attribute 'has_previous_state'
```

Traceback points to `transformers/models/qwen3_5/modeling_qwen3_5.py:525`:
```python
use_precomputed_states = (
    cache_params is not None
    and cache_params.has_previous_state   # <-- fails here
    and seq_len == 1
    ...
)
```

The Qwen3.5 model is a hybrid (Transformer+Mamba-like) architecture that defines
its own `Qwen3_5DynamicCache` with a `has_previous_state` property, but the
benchmark harness creates a `StaticCache` (standard HF cache) which does not
have this attribute. This is a model-loader / transformers version incompatibility
(transformers==5.2.0) that must be fixed in tt-forge-models, not here.

This skill does not patch model definitions or loader code.

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach compilation

## Files changed
- tests/benchmark/test_llms.py (test function added)

## tt-forge-models submodule
no change — submodule at 8637301436
