loader_path: third_party.tt_forge_models.xlm_roberta.causal_lm.pytorch.loader
variant_id: Tiny
arch: p150
status: DONE_FAIL
test_function: test_xlm_roberta_tiny
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
failure_reason: "loader uses _tokenizer private attribute instead of public tokenizer property required by benchmark harness (AttributeError: 'ModelLoader' object has no attribute 'tokenizer')"

# Benchmark added: test_xlm_roberta_tiny

## Test
tests/benchmark/test_llms.py::test_xlm_roberta_tiny

## Model
- HF name:    optimum-intel-internal-testing/tiny-xlm-roberta
- Loader:     third_party.tt_forge_models.xlm_roberta.causal_lm.pytorch.loader
- Variant:    ModelVariant.TINY

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

## Failure Details
The benchmark run failed immediately after model weight loading with:

    AttributeError: 'ModelLoader' object has no attribute 'tokenizer'. Did you mean: '_tokenizer'?

The XLM-RoBERTa loader in `third_party/tt_forge_models/xlm_roberta/causal_lm/pytorch/loader.py`
stores the tokenizer as `self._tokenizer` (private attribute) rather than the public
`self.tokenizer` attribute expected by the benchmark harness
(`tests/benchmark/benchmarks/llm_benchmark.py:80: tokenizer = model_loader.tokenizer`).

Other loaders (e.g. `qwen_3`, `llama`) correctly expose `self.tokenizer` as a public
attribute. The fix belongs in the `tt-forge-models` repository — the loader must be
updated to expose a public `tokenizer` property. No edits to files under
`third_party/tt_forge_models/` are within scope of this skill.

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test failed before compilation)
Achieved vs top_perf_samples_per_sec: N/A

### System
- arch: N/A

### Roofline
- bound: N/A
- top_perf_samples_per_sec: N/A
- top_perf_time_ms: N/A

## Files changed
- tests/benchmark/test_llms.py (test_xlm_roberta_tiny added)

## tt-forge-models submodule
no change
