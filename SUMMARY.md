loader_path: third_party.tt_forge_models.tiny_gpt2_lm_head.causal_lm.pytorch.loader
variant_id: tiny_gpt2_lm_head
arch: p150
status: DONE_FAIL
test_function: test_tiny_gpt2_lm_head
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
failure_reason: "model config has incorrect num_key_value_heads=2 (GPT-2 does not support GQA; actual attention uses n_head=4 heads for K/V), causing StaticCache shape mismatch: cache allocated with [32,2,128,2] but model produces keys [32,4,1,2], resulting in index_copy_ RuntimeError in golden decode"

# Benchmark added: test_tiny_gpt2_lm_head

## Test
tests/benchmark/test_llms.py::test_tiny_gpt2_lm_head

## Model
- HF name:    trl-internal-testing/tiny-GPT2LMHeadModel
- Loader:     third_party.tt_forge_models.tiny_gpt2_lm_head.causal_lm.pytorch.loader
- Variant:    ModelVariant.TINY_GPT2_LM_HEAD

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
The model `trl-internal-testing/tiny-GPT2LMHeadModel` has `num_key_value_heads=2`
in its HuggingFace config.json, but standard GPT-2 does not support GQA (Grouped Query
Attention). The actual attention computation uses `n_head=4` for all of Q, K, and V.

The benchmark harness calls `init_static_cache` which reads `num_key_value_heads=2`
from the config and pre-allocates the KV cache with shape `[32, 2, 128, 2]`
(batch=32, heads=2, max_len=128, head_dim=2).

During the golden (CPU) decode pass, the GPT-2 model produces key_states with
shape `[32, 4, 1, 2]` (4 heads, not 2). The `StaticCache.update()` call then
fails at `self.keys.index_copy_(2, cache_position, key_states)` with:

  RuntimeError: index_copy_(): Source/destination tensor must have same slice shapes.
  Destination slice shape: 32 2 2 at dimension 2 and source slice shape: 32 4 2 at dimension 0.

This is a model configuration bug in the loader's underlying model:
`num_key_value_heads` should equal `n_head=4` (or be removed) for standard GPT-2.
The fix belongs in `third_party/tt_forge_models/tiny_gpt2_lm_head/` (the loader
should override the config to set `num_key_value_heads = config.n_head` before
loading), which is out of scope for this skill.

Submodule HEAD at time of failure: 30c94449f5

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach hardware execution

## Files changed
- tests/benchmark/test_llms.py (test function added)
- SUMMARY.md

## tt-forge-models submodule
no change
