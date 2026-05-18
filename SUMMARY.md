loader_path: third_party.tt_forge_models.huginn_0125.causal_lm.pytorch.loader
variant_id: 0125
arch: n150
status: DONE_FAIL
test_function: test_huginn_0125
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
failure_reason: "IndexError: list index out of range in raven_modeling_minimal.py:479 - Huginn recurrent depth architecture uses block_idx as KV cache layer index but block_idx exceeds n_layers because core_block is applied multiple times in iterate_forward; model-level incompatibility between trust_remote_code raven_modeling_minimal.py and transformers DynamicCache"

# Benchmark added: test_huginn_0125

## Test
tests/benchmark/test_llms.py::test_huginn_0125

## Model
- HF name:    tomg-group-umd/huginn-0125
- Loader:     third_party.tt_forge_models.huginn_0125.causal_lm.pytorch.loader
- Variant:    ModelVariant.HUGINN_0125

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

## Failure details

Both `--num-layers 1` and full model runs fail with:

```
IndexError: list index out of range
  raven_modeling_minimal.py:682: in forward
      x, ... = self.iterate_forward(...)
  raven_modeling_minimal.py:751: in iterate_forward
      x, block_idx = self.core_block_forward(...)
  raven_modeling_minimal.py:777: in core_block_forward
      x = block(x, freqs_cis, block_idx, mask, past_key_values)
  raven_modeling_minimal.py:533: in forward (block)
      attn_out = self.attn(self.norm_1(x), freqs_cis, step_idx, mask, past_key_values)
  raven_modeling_minimal.py:479: in forward (attn)
      k, v = past_key_values.update(k, v, block_idx)
  transformers/cache_utils.py:792: in update
      keys, values = self.layers[layer_idx].update(...)
  IndexError: list index out of range
```

Root cause: The model uses a "recurrent depth" architecture (Raven) where
`core_block` is applied multiple times in `iterate_forward`. The `block_idx`
increments monotonically across iterations, so it quickly exceeds the number
of unique transformer layers the KV cache was initialized for. The fix would
require the model's `raven_modeling_minimal.py` (downloaded via trust_remote_code)
to use a modular index for KV cache lookups, or allocate the cache with
`n_layers * num_iterations` slots. This is a model-level fix outside this skill's scope.

## Decode roofline (first decode graph, single-chip)
N/A - model failed before compilation

## Files changed
- tests/benchmark/test_llms.py (test function added)

## tt-forge-models submodule
no change
