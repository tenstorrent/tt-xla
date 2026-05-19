loader_path: third_party.tt_forge_models.olmo3_recurrent_adapter_sft_cot.causal_lm.pytorch.loader
variant_id: recurrent_adapter_sft_cot
arch: p150
status: DONE_FAIL
test_function: test_olmo3_recurrent_adapter_sft_cot
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
failure_reason: "RecurrentAdapterConfig lacks required attributes (hidden_size, num_attention_heads, num_hidden_layers, head_dim); model sets _supports_cache_class=False, incompatible with StaticCache-based benchmark harness"

# Benchmark added: test_olmo3_recurrent_adapter_sft_cot

## Test
tests/benchmark/test_llms.py::test_olmo3_recurrent_adapter_sft_cot

## Model
- HF name:    hanseungwook/olmo3-recurrent-adapter-sft-cot
- Loader:     third_party.tt_forge_models.olmo3_recurrent_adapter_sft_cot.causal_lm.pytorch.loader
- Variant:    ModelVariant.Olmo3_Recurrent_Adapter_SFT_CoT

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8 (default)
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Failure Analysis

The test fails at `init_static_cache` in `tests/benchmark/llm_utils/decode_utils.py:125`:

```
AttributeError: 'RecurrentAdapterConfig' object has no attribute 'hidden_size'
```

**Root cause:** `RecurrentAdapterConfig` (the custom config for `hanseungwook/olmo3-recurrent-adapter-sft-cot`)
is a non-standard config that wraps OLMo-3-7B and adds recurrent adapter layers. It does not expose any
of the standard transformer config attributes required by the benchmark harness's `StaticCache`
initialization:

| Attribute             | Value       |
|-----------------------|-------------|
| `hidden_size`         | NOT FOUND   |
| `num_attention_heads` | NOT FOUND   |
| `num_hidden_layers`   | NOT FOUND   |
| `head_dim`            | NOT FOUND   |
| `num_key_value_heads` | NOT FOUND   |

Additionally, the `RecurrentAdapterModel` class explicitly sets `_supports_cache_class = False`,
meaning it does not support `StaticCache` at all. The model uses a `List[Tuple[Tensor, Tensor]]`
format for `past_key_values` and has `use_cache=False` as its default — fundamentally different
from the standard decode loop in the benchmark harness.

Supporting this model would require restructuring the benchmark flow to handle non-StaticCache
architectures (detecting `_supports_cache_class=False`, providing an empty list as the initial
cache, and running prefill+decode without `StaticCache`). This is out of scope for this skill.

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not reach compilation)
Achieved vs top_perf_samples_per_sec: N/A

## Files changed
- tests/benchmark/test_llms.py (test function added)
- SUMMARY.md

## tt-forge-models submodule
no change
