loader_path: third_party.tt_forge_models.gemma3_npc_1b_float16_i1_gguf.causal_lm.pytorch.loader
variant_id: 1B_FLOAT16_I1_Q4_K_M_GGUF
arch: p150
status: DONE_FAIL
test_function: test_gemma3_npc_1b_float16_i1_q4_k_m_gguf
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
failure_reason: "KeyError: 'sliding_attention' in transformers Gemma3 model forward pass (modeling_gemma3.py:589) — GGUF-loaded model config missing sliding_attention rotary embeddings; fails on CPU before reaching TT device"

# Benchmark added: gemma3_npc_1b_float16_i1_q4_k_m_gguf

## Test
tests/benchmark/test_llms.py::test_gemma3_npc_1b_float16_i1_q4_k_m_gguf

## Model
- HF name:    mradermacher/Gemma3NPC-1b-float16-i1-GGUF
- Loader:     third_party.tt_forge_models.gemma3_npc_1b_float16_i1_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.GEMMA3_NPC_1B_FLOAT16_I1_Q4_K_M_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test failed before TT device run)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details
The test fails with `KeyError: 'sliding_attention'` inside the HuggingFace
transformers Gemma3 model forward pass (`modeling_gemma3.py:589`) during
the CPU golden reference run — before the model is ever dispatched to the
TT device. The error occurs because the GGUF file (Gemma3NPC-1b-float16.i1-Q4_K_M.gguf)
loads a model configuration that includes layers with `attention_type='sliding_attention'`,
but the `position_embeddings` dict constructed during the forward pass only
contains the `'global_attention'` key. This is a model-loader incompatibility
between the GGUF format and the transformers==5.2.0 Gemma3 implementation.

This is out of scope for the benchmark test skill — the fix belongs in the
tt-forge-models loader or the transformers library. No changes to files
under `third_party/tt_forge_models/` have been made.

Traceback summary:
  tests/benchmark/benchmarks/llm_benchmark.py:406: in benchmark_llm_torch_xla
      cpu_prefill_logits, _ = generate_and_benchmark(...)
  tests/benchmark/llm_utils/decode_utils.py:322: in generate_and_benchmark
      output = model(**input_args)
  transformers/models/gemma3/modeling_gemma3.py:589: in forward
      position_embeddings=position_embeddings[decoder_layer.attention_type],
  KeyError: 'sliding_attention'

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach TT device

## Files changed
- tests/benchmark/test_llms.py (test function added)

## tt-forge-models submodule
no change
