loader_path: third_party.tt_forge_models.ci_random_gpt2_350m.causal_lm.pytorch.loader
variant_id: Default
arch: p150
status: DONE_FAIL
test_function: test_ci_random_gpt2_350m
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
failure_reason: "loader passes use_cache=False to GPT2LMHeadModel.from_pretrained() which is rejected by transformers 5.2 (TypeError: GPT2LMHeadModel.__init__() got an unexpected keyword argument 'use_cache') — fix belongs in tt-forge-models loader"

# Benchmark added: ci_random_gpt2_350m

## Test
tests/benchmark/test_llms.py::test_ci_random_gpt2_350m

## Model
- HF name:    hyper-accel/ci-random-gpt2-350m
- Loader:     third_party.tt_forge_models.ci_random_gpt2_350m.causal_lm.pytorch.loader
- Variant:    ModelVariant.DEFAULT ("Default")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test failed before reaching inference)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details
The loader at `third_party/tt_forge_models/ci_random_gpt2_350m/causal_lm/pytorch/loader.py`
passes `use_cache=False` as a model kwarg to `GPT2LMHeadModel.from_pretrained()`:

    model_kwargs = {"use_cache": False}
    ...
    model = GPT2LMHeadModel.from_pretrained(pretrained_model_name, **model_kwargs)

In transformers 5.2, `GPT2LMHeadModel.__init__()` only accepts `(self, config)` and
no longer forwards extra kwargs to the config. This raises:

    TypeError: GPT2LMHeadModel.__init__() got an unexpected keyword argument 'use_cache'

Transformers version: 5.2.0
Submodule HEAD: e496eed1c3 (ci_random_gpt2_350m/causal_lm/pytorch/loader.py)

The fix must be applied in the tt-forge-models repo. The loader should set
`use_cache` on the config object directly (via `AutoConfig.from_pretrained` +
`config.use_cache = False`) or omit the kwarg entirely since the benchmark
harness controls KV cache via StaticCache.

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach compilation or inference

## Files changed
- tests/benchmark/test_llms.py (test function added, will fail until loader is fixed)

## tt-forge-models submodule
no change — submodule at e496eed1c3
