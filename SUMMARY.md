loader_path: third_party.tt_forge_models.deepseek_ocr_gguf.causal_lm.pytorch.loader
variant_id: CUDA_Q4_0
arch: p150
status: DONE_FAIL
test_function: test_deepseek_ocr_gguf_cuda_q4_0
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
failure_reason: "DeepseekOCRForCausalLM.forward() lacks KV-cache support; decode-time attention runs on seq=1, violating sdpa_decode hardware constraint k_chunk_size%32==0 (got 2). Model needs KV-cache refactoring in tt-forge-models."

# Benchmark added: test_deepseek_ocr_gguf_cuda_q4_0

## Test
tests/benchmark/test_llms.py::test_deepseek_ocr_gguf_cuda_q4_0

## Model
- HF name:    NexaAI/DeepSeek-OCR-GGUF-CUDA
- Loader:     third_party.tt_forge_models.deepseek_ocr_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.DEEPSEEK_OCR_CUDA_Q4_0

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (FAILED)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details

The model's `DeepseekOCRForCausalLM.forward()` does not implement KV-cache
support. During decode, the harness calls the model with `input_ids` of shape
`[batch_size, 1]` (one token at a time), causing the self-attention to compute
over only 1 key/value token. The TT hardware's `sdpa_decode` operation
requires `k_chunk_size % 32 == 0`, but with a KV sequence length of 1 the
calculated `k_chunk_size` is 2, which violates this constraint.

This occurs at both `optimization_level=2` and `optimization_level=0`.

**Root cause (in tt-forge-models):** `DeepseekOCRAttention.forward()` uses
`torch.nn.functional.scaled_dot_product_attention` directly without any
`past_key_values` accumulation. The model needs KV-cache refactoring to
support the benchmark harness's decode loop.

Two general infrastructure fixes were made to `tests/benchmark/` to handle
custom GGUF model configs that predate these patterns (both are loader-agnostic
and benefit future models with the same design):

1. `benchmarks/llm_benchmark.py`: Added fallback `_load_tokenizer()` call in
   `setup_model_and_tokenizer()` when `model_loader.tokenizer` is None after
   `load_model()`.

2. `llm_utils/decode_utils.py`: Monkey-patch `get_text_config()` onto custom
   config dataclasses before calling `StaticCache.__init__()`, since
   `transformers.StaticCache` requires this method but plain `@dataclass`
   configs don't inherit it from `PretrainedConfig`.

3. `benchmarks/llm_benchmark.py`: Guard `get_weight_dtype_config_path()` call
   with `hasattr()` check (matching the pattern already used in
   `tests/runner/testers/torch/dynamic_torch_model_tester.py`).

## Decode roofline (first decode graph, single-chip)
N/A — test did not complete

## Files changed
- tests/benchmark/test_llms.py (added test_deepseek_ocr_gguf_cuda_q4_0)
- tests/benchmark/benchmarks/llm_benchmark.py (tokenizer fallback + weight dtype guard)
- tests/benchmark/llm_utils/decode_utils.py (get_text_config monkey-patch)

## tt-forge-models submodule
no change
