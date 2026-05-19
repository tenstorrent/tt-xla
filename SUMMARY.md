loader_path: third_party.tt_forge_models.mradermacher_qwen3_5_4b_claude_4_6_opus_reasoning_distill_heretic_v3_i1_gguf.causal_lm.pytorch.loader
variant_id: 4B_Claude_4.6_Opus_Reasoning_Distill_heretic_v3_i1_GGUF
arch: p150
status: DONE_FAIL
test_function: test_mradermacher_qwen3_5_4b_heretic_v3_i1_gguf
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
failure_reason: "loader bug: AutoTokenizer.from_pretrained fails with ValueError: Unrecognized model identifier: qwen35 — the loader's _patch_qwen35_support() does not register qwen35 in AutoConfig's model-type registry, only in GGUF_SUPPORTED_ARCHITECTURES and GGUF_TO_TRANSFORMERS_MAPPING; fix belongs in tt-forge-models loader"

# Benchmark added: test_mradermacher_qwen3_5_4b_heretic_v3_i1_gguf

## Test
tests/benchmark/test_llms.py::test_mradermacher_qwen3_5_4b_heretic_v3_i1_gguf

## Model
- HF name:    mradermacher/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distill-heretic-v3-i1-GGUF
- Loader:     third_party.tt_forge_models.mradermacher_qwen3_5_4b_claude_4_6_opus_reasoning_distill_heretic_v3_i1_gguf.causal_lm.pytorch.loader
- Variant:    MRADERMACHER_QWEN3_5_4B_CLAUDE_4_6_OPUS_REASONING_DISTILL_HERETIC_V3_I1_GGUF

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: bfp_bf8
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Measured (full model, defaults)
- Sample per second:  N/A (test failed before model ran)
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Failure Details

The test fails immediately in `_load_tokenizer` at the very first bring-up run
(`--num-layers 1 --max-output-tokens 3`) with:

```
ValueError: Unrecognized model identifier: qwen35. Should contain one of ... qwen3_5 ...
```

Traceback:
  tests/benchmark/test_llms.py::test_mradermacher_qwen3_5_4b_heretic_v3_i1_gguf
  → test_llm (test_llms.py:135)
  → benchmark_llm_torch_xla (llm_benchmark.py:358)
  → setup_model_and_tokenizer (llm_benchmark.py:72)
  → model_loader.load_model (loader.py:179)
  → self._load_tokenizer (loader.py:167)
  → AutoTokenizer.from_pretrained (transformers 5.2.0)
  → AutoConfig.for_model(**config_dict) ← fails here

Root cause: The GGUF file's architecture is "qwen35" (no underscore). The loader
has a `_patch_qwen35_support()` function that patches `GGUF_SUPPORTED_ARCHITECTURES`
and `GGUF_TO_TRANSFORMERS_MAPPING`, and wraps `load_gguf_checkpoint` to rewrite
`model_type: "qwen35"` → `model_type: "qwen3"` in the checkpoint dict. However,
the patch does NOT register "qwen35" in `AutoConfig`'s internal model-type
registry (`CONFIG_MAPPING` / `configuration_auto.py`), so when
`AutoTokenizer.from_pretrained(..., gguf_file=...)` calls
`AutoConfig.for_model(model_type="qwen35", ...)` internally, it raises a
ValueError.

Transformers 5.2.0 is installed. The supported identifier is `qwen3_5`
(with underscore), not `qwen35`.

Fix required in tt-forge-models: extend `_patch_qwen35_support()` to also
register "qwen35" as an alias in `AutoConfig`'s model-type registry, e.g.:
  from transformers.models.auto.configuration_auto import CONFIG_MAPPING
  CONFIG_MAPPING.register("qwen35", CONFIG_MAPPING["qwen3"])

This is a loader fix; editing files under third_party/tt_forge_models/ is out
of scope for this skill.

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach compilation or inference.

## Files changed
- tests/benchmark/test_llms.py (test function added, DONE_FAIL)

## tt-forge-models submodule
no change — submodule HEAD: 8bbda16005
