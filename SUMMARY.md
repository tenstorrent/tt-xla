loader_path: third_party.tt_forge_models.holy_fox_qwen3_5_0_8b_jp_gguf.causal_lm.pytorch.loader
variant_id: 0.8B_JP_GGUF
arch: p150
status: DONE_FAIL
test_function: test_holy_fox_qwen3_5_0_8b_jp_gguf
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
failure_reason: "transformers 5.2.0 does not support GGUF architecture 'qwen35'; ValueError: GGUF model with architecture qwen35 is not supported yet"

# Benchmark added: holy_fox_qwen3_5_0_8b_jp_gguf

## Test
tests/benchmark/test_llms.py::test_holy_fox_qwen3_5_0_8b_jp_gguf

## Model
- HF name:    mmnga-o/Holy-fox-Qwen3.5-0.8B-JP-gguf
- Loader:     third_party.tt_forge_models.holy_fox_qwen3_5_0_8b_jp_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.HOLY_FOX_QWEN3_5_0_8B_JP_GGUF ("0.8B_JP_GGUF")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8"
- batch_size:                32
- input_sequence_length:     128
- required_pcc:              0.94

## Failure
The test fails immediately when loading the model tokenizer:

```
ValueError: GGUF model with architecture qwen35 is not supported yet.
```

Stack trace:
```
transformers/models/auto/tokenization_auto.py:620: in from_pretrained
    config_dict = load_gguf_checkpoint(gguf_path, return_tensors=False)["config"]
transformers/modeling_gguf_pytorch_utils.py:478: in load_gguf_checkpoint
    raise ValueError(f"GGUF model with architecture {architecture} is not supported yet.")
```

The installed transformers version (5.2.0) does not include `qwen35` in its
`GGUF_SUPPORTED_ARCHITECTURES` list. The supported set includes `qwen3` but not
`qwen35` (which is the GGUF architecture identifier used by Qwen 3.5 models).
A newer transformers release that adds `qwen35` GGUF support is required.

This is a dependency-level incompatibility. Fixing it requires either:
1. Upgrading the `transformers` package to a version that supports `qwen35` GGUF.
2. Patching `transformers/modeling_gguf_pytorch_utils.py` to map `qwen35` → `qwen3`.

Neither fix is within the scope of this skill (no edits to third_party loaders or
venv packages). The loader and test function are correct; the environment needs updating.

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           p150

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test did not run)
Achieved vs top_perf_samples_per_sec: N/A

## Files changed
- tests/benchmark/test_llms.py (test_holy_fox_qwen3_5_0_8b_jp_gguf added at line 1094)
- SUMMARY.md

## tt-forge-models submodule
no change
