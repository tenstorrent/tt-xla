loader_path: third_party.tt_forge_models.gemma3_12b_cybersecurity_gguf.causal_lm.pytorch.loader
variant_id: 12B_cybersecurity_GGUF
arch: p150
status: DONE_FAIL
test_function: test_gemma3_12b_cybersecurity_gguf
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
failure_reason: "KeyError: 'sliding_attention' in transformers 5.2.0 modeling_gemma3.py:589 during CPU golden run — GGUF attention_type mismatch prevents model from running"

# Benchmark added: test_gemma3_12b_cybersecurity_gguf

## Test
tests/benchmark/test_llms.py::test_gemma3_12b_cybersecurity_gguf

## Model
- HF name:    jayanthanc/gemma3-12b-cybersecurity-GGUF
- Loader:     third_party.tt_forge_models.gemma3_12b_cybersecurity_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GEMMA3_12B_CYBERSECURITY_GGUF (= "12B_cybersecurity_GGUF")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Failure
The test fails during the CPU golden run (before any TT device compilation) with:

    KeyError: 'sliding_attention'

in `transformers/models/gemma3/modeling_gemma3.py:589`:

    position_embeddings=position_embeddings[decoder_layer.attention_type],

The GGUF file encodes some layers with `attention_type='sliding_attention'`, but
transformers 5.2.0's Gemma3 model builds a position_embeddings dict that does not
include a 'sliding_attention' key.  This is a compatibility issue between the GGUF
file and the installed transformers version — outside the scope of this skill to fix.

Full traceback:
    tests/benchmark/test_llms.py::test_gemma3_12b_cybersecurity_gguf
    tests/benchmark/benchmarks/llm_benchmark.py:406 benchmark_llm_torch_xla
    tests/benchmark/llm_utils/decode_utils.py:322 generate_and_benchmark
    tests/benchmark/llm_utils/decode_utils.py:58 forward
    transformers/models/gemma3/modeling_gemma3.py:589 forward
    KeyError: 'sliding_attention'

## Measured (full model, defaults)
- Sample per second:  N/A (model failed before TT run)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150 (blackhole-p150b)

## Decode roofline (first decode graph, single-chip)
N/A — no perf JSON produced (model failed before compilation)

## Files changed
- tests/benchmark/test_llms.py (test function added, left in place for future use)

## tt-forge-models submodule
no change
