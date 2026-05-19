loader_path: third_party.tt_forge_models.tiny_gpt_neox.causal_lm.pytorch.loader
variant_id: tiny
arch: p150
status: DONE_FAIL
test_function: test_tiny_gpt_neox
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
failure_reason: "model config declares num_key_value_heads=2 (GQA) but QKV weight [24, 8] implies num_kv_heads=4 (full MHA); StaticCache initialized with num_kv_heads=2 from config but model forward produces 4 KV heads, causing index_copy_ shape mismatch in CPU reference forward pass — fix requires correcting trl-internal-testing/tiny-GPTNeoXForCausalLM model definition or loader config override"

# Benchmark added: test_tiny_gpt_neox

## Test
tests/benchmark/test_llms.py::test_tiny_gpt_neox

## Model
- HF name:    trl-internal-testing/tiny-GPTNeoXForCausalLM
- Loader:     third_party.tt_forge_models.tiny_gpt_neox.causal_lm.pytorch.loader
- Variant:    ModelVariant.TINY_GPT_NEOX ("tiny")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (DONE_FAIL)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure details
The test fails in the CPU reference forward pass before any TT-device code runs.

Root cause: `trl-internal-testing/tiny-GPTNeoXForCausalLM` has an inconsistent
model definition:
- `config.num_key_value_heads = 2` (GQA — 4 query heads, 2 KV heads)
- Actual QKV weight shape = `[24, 8]` = `3 * num_attention_heads * head_dim = 3*4*2`
  (full MHA format — implies 4 KV heads, same as Q heads)

When transformers 5.2.0 processes this model, it:
1. Initializes `StaticCache` with `num_kv_heads=2` (from config)
2. Runs QKV projection → produces 4 KV head tensors (from weight shape)
3. Calls `StaticCacheLayer.update` → `index_copy_(dim=2, ...)` fails because
   destination cache slice `[32, 2, 2]` != source key_states slice `[32, 4, 2]`

```
RuntimeError: index_copy_(): Source/destination tensor must have same slice shapes.
Destination slice shape: 32 2 2 at dimension 2 and source slice shape: 32 4 2 at dimension 0.
```

The forward pass works fine with batch_size=1 and DynamicCache (default `use_cache=True`
in bare `model()` call), but fails when the benchmark harness sets up a StaticCache
with `num_kv_heads=2` at `batch_size=32`.

Fix belongs in tt-forge-models: either update the loader to override
`num_key_value_heads=4` in the model config before StaticCache initialization,
or correct the model's config.json to match the actual weights.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (DONE_FAIL — test never reached TT device)

## Files changed
- tests/benchmark/test_llms.py (new test_tiny_gpt_neox function added)

## tt-forge-models submodule
no change — submodule at 4f9cc054aa (93218a34fc9fc6a671e0e41101da470c80891b2a)
