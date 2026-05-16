loader_path: third_party.tt_forge_models.albert_wesker_gguf.causal_lm.pytorch.loader
variant_id: 1B_GGUF
arch: n150
status: DONE_FAIL
test_function: test_albert_wesker_1b_gguf
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: 2
trace_enabled: true
experimental_weight_dtype: "bfp_bf8"
failure_reason: "KeyError: 'sliding_attention' in Gemma3 forward pass - model/transformers version incompatibility; cannot fix without editing loader"

# Benchmark added: test_albert_wesker_1b_gguf

## Test
tests/benchmark/test_llms.py::test_albert_wesker_1b_gguf

## Model
- HF name:    mradermacher/Albert_Wesker-1B-GGUF
- Loader:     third_party.tt_forge_models.albert_wesker_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.ALBERT_WESKER_1B_GGUF (value: "1B_GGUF")

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
- Hardware:           n150 (n300 wormhole_b0)

## Failure details
The test failed during model bring-up (--num-layers 1 --max-output-tokens 3) with:

    KeyError: 'sliding_attention'
      File "transformers/models/gemma3/modeling_gemma3.py", line 589, in forward
        position_embeddings=position_embeddings[decoder_layer.attention_type]

The Albert Wesker 1B GGUF model is based on Gemma3, which uses mixed attention types
("global" and "sliding_attention"). The current transformers version does not properly
populate the position_embeddings dict with the sliding_attention key during the forward
pass, causing a KeyError at runtime. This is a model definition / transformers version
incompatibility in the loader - fixing it requires changes under third_party/tt_forge_models/,
which is out of scope for this skill.

## Decode roofline (first decode graph, single-chip)
N/A - test did not reach compilation or execution.

## Files changed
- tests/benchmark/test_llms.py (test function added)

## tt-forge-models submodule
no change
