loader_path: third_party.tt_forge_models.cosmos_reason1_gguf.causal_lm.pytorch.loader
variant_id: 7B_GGUF
arch: p150
status: DONE_FAIL
test_function: test_cosmos_reason1_7b_gguf
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
failure_reason: "transformers does not support qwen2vl GGUF architecture: ValueError: GGUF model with architecture qwen2vl is not supported yet."

# Benchmark added: cosmos_reason1_7b_gguf

## Test
tests/benchmark/test_llms.py::test_cosmos_reason1_7b_gguf

## Model
- HF name:    unsloth/Cosmos-Reason1-7B-GGUF
- Loader:     third_party.tt_forge_models.cosmos_reason1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.COSMOS_REASON1_7B_GGUF (7B_GGUF)

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

## Failure Details
The test failed at model loading time. The loader attempts to load the GGUF
file `Cosmos-Reason1-7B-Q4_K_M.gguf` from HuggingFace repo
`unsloth/Cosmos-Reason1-7B-GGUF`. The GGUF file internally identifies its
architecture as `qwen2vl`, which is a vision-language model architecture.

The current version of the `transformers` library in this environment does
not support GGUF loading for the `qwen2vl` architecture, raising:

    ValueError: GGUF model with architecture qwen2vl is not supported yet.

This is raised in `transformers/modeling_gguf_pytorch_utils.py:478` during
tokenizer initialization. The fix requires either a newer transformers that
supports qwen2vl GGUF, or using an alternate loading path in the loader (which
is out of scope for this skill — loader changes belong in tt-forge-models).

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (model did not load)
Achieved vs top_perf_samples_per_sec: N/A

### System
All values N/A — model failed before compilation.

## Files changed
- tests/benchmark/test_llms.py (test_cosmos_reason1_7b_gguf added)

## tt-forge-models submodule
no change
