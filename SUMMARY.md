loader_path: third_party.tt_forge_models.toolace_2_llama_3_1.causal_lm.pytorch.loader
variant_id: Default
arch: p150
status: DONE_FAIL
test_function: test_toolace_2_llama_3_1_8b
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
failure_reason: "loader bug: LlamaForCausalLM.__init__() got an unexpected keyword argument 'use_cache' — loader passes use_cache=False directly via from_pretrained kwargs, incompatible with current transformers version; fix belongs in tt-forge-models repo at submodule HEAD e51150dc10"

# Benchmark added: test_toolace_2_llama_3_1_8b

## Test
tests/benchmark/test_llms.py::test_toolace_2_llama_3_1_8b

## Model
- HF name:    Team-ACE/ToolACE-2-Llama-3.1-8B
- Loader:     third_party.tt_forge_models.toolace_2_llama_3_1.causal_lm.pytorch.loader
- Variant:    Default (ModelVariant.TOOLACE_2_LLAMA_3_1_8B)

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Failure
The loader in `third_party/tt_forge_models/toolace_2_llama_3_1/causal_lm/pytorch/loader.py`
sets `model_kwargs["use_cache"] = False` and passes it directly to
`AutoModelForCausalLM.from_pretrained(...)`. The current transformers version
passes this as a constructor keyword argument, but `LlamaForCausalLM.__init__()`
does not accept `use_cache` as a direct kwarg.

```
TypeError: LlamaForCausalLM.__init__() got an unexpected keyword argument 'use_cache'
```

Submodule HEAD at time of failure: e51150dc10

This is a loader-side bug. The fix belongs in the tt-forge-models repo.
Per skill rules, no files under `third_party/tt_forge_models/` were modified.

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150 (Blackhole p300c)

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach compilation/execution stage.

## Files changed
- tests/benchmark/test_llms.py (added test_toolace_2_llama_3_1_8b)

## tt-forge-models submodule
no change — submodule remains at e51150dc10
