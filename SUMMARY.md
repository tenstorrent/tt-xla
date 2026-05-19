loader_path: third_party.tt_forge_models.testing_gpt_oss.causal_lm.pytorch.loader
variant_id: Default
arch: p150
status: DONE_FAIL
test_function: test_testing_gpt_oss
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
failure_reason: "TypeError: GptOssForCausalLM.__init__() got an unexpected keyword argument 'use_cache' — loader passes use_cache=False but model does not accept it"

# Benchmark added: test_testing_gpt_oss

## Test
tests/benchmark/test_llms.py::test_testing_gpt_oss

## Model
- HF name:    np-cr/testing-gpt-oss
- Loader:     third_party.tt_forge_models.testing_gpt_oss.causal_lm.pytorch.loader
- Variant:    ModelVariant.TESTING_GPT_OSS (Default)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
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

## Failure details
The bring-up run at `--num-layers 1 --max-output-tokens 3` failed immediately
during model loading with:

```
TypeError: GptOssForCausalLM.__init__() got an unexpected keyword argument 'use_cache'
```

The loader's `load_model()` sets `model_kwargs["use_cache"] = False` and
passes it to `AutoModelForCausalLM.from_pretrained()`, but the custom
`GptOssForCausalLM` model does not accept `use_cache` as a constructor
argument. This is a bug in the loader that must be fixed in the
tt-forge-models repo (cannot be patched from tt-xla per skill policy).

Submodule HEAD at time of failure: 9041a7b7db

## Decode roofline (first decode graph, single-chip)
N/A — model failed to load

## Files changed
- tests/benchmark/test_llms.py

## tt-forge-models submodule
no change
