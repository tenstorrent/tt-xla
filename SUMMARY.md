loader_path: third_party.tt_forge_models.gemma3_orthogonal_reflection_gguf.causal_lm.pytorch.loader
variant_id: 12B_IT_Orthogonal_Reflection_GGUF
arch: p150
status: DONE_FAIL
test_function: test_gemma3_orthogonal_reflection_12b
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
failure_reason: "KeyError: 'sliding_attention' in transformers/models/gemma3/modeling_gemma3.py — GGUF model incompatibility with installed transformers version"

# Benchmark added: test_gemma3_orthogonal_reflection_12b

## Test
tests/benchmark/test_llms.py::test_gemma3_orthogonal_reflection_12b

## Model
- HF name:    mradermacher/gemma-3-12b-it-orthogonal-reflection-bounded-ablation-v3-12B-i1-GGUF
- Loader:     third_party.tt_forge_models.gemma3_orthogonal_reflection_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GEMMA_3_12B_IT_ORTHOGONAL_REFLECTION_GGUF

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

## Failure
The test failed during the bring-up run (--num-layers 1, --max-output-tokens 3) with:

    KeyError: 'sliding_attention'

occurring in `transformers/models/gemma3/modeling_gemma3.py:589`:

    position_embeddings=position_embeddings[decoder_layer.attention_type]

The model's decoder layer reports `attention_type='sliding_attention'` but the position_embeddings dict
computed by the GGUF-loaded Gemma 3 model does not include that key. This is a compatibility issue
between the GGUF model file and the installed `transformers` version in the venv. The error is in
the model's forward pass, not in the benchmark harness, and cannot be fixed at the test level.

The test function has been added to `tests/benchmark/test_llms.py` with a `# FAILED:` comment
following the established pattern (see `test_gemma_1_1_7b`, `test_phi3_mini`).

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach the compilation/execution stage

## Files changed
- tests/benchmark/test_llms.py

## tt-forge-models submodule
no change
