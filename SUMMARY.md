loader_path: third_party.tt_forge_models.zuzett_qwen3_5_4b_heretic_gguf.causal_lm.pytorch.loader
variant_id: 4B_Heretic_GGUF
arch: p150
status: DONE_FAIL
test_function: test_zuzett_qwen3_5_4b_heretic_gguf
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
failure_reason: "GGUF architecture qwen35 not supported in transformers 5.2.0; AutoTokenizer.from_pretrained(gguf_file=...) raises ValueError: GGUF model with architecture qwen35 is not supported yet"

# Benchmark added: test_zuzett_qwen3_5_4b_heretic_gguf

## Test
tests/benchmark/test_llms.py::test_zuzett_qwen3_5_4b_heretic_gguf

## Model
- HF name:    ZuzeTt/Qwen3.5-4B-heretic-GGUF
- Loader:     third_party.tt_forge_models.zuzett_qwen3_5_4b_heretic_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.QWEN3_5_4B_HERETIC_GGUF ("4B_Heretic_GGUF")

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

Bring-up failed at Step 3 with:

```
ValueError: GGUF model with architecture qwen35 is not supported yet.
```

The loader's `_load_tokenizer` calls `AutoTokenizer.from_pretrained` with
`gguf_file="Qwen3.5-4B-heretic-imatrix-Q4_K_M.gguf"`. Internally this
calls `transformers.modeling_gguf_pytorch_utils.load_gguf_checkpoint`, which
reads the GGUF file's `general.architecture` field (`qwen35`) and raises
because `qwen35` is not in `GGUF_SUPPORTED_ARCHITECTURES` for transformers
5.2.0. The supported list includes `qwen3` and `qwen3_moe` but not `qwen35`.

The loader has a `_zuzett_load_gguf` workaround for the model *weights*
(monkey-patching `_gguf_utils.load_gguf_checkpoint` inside `load_model`),
but the tokenizer path (`_load_tokenizer`) is called before that context is
entered and uses the default transformers loader directly.

Fixing this requires either:
1. Updating transformers to a version that adds `qwen35` support, or
2. Extending the loader's monkey-patch to also cover tokenizer loading.

Both options require changes in `third_party/tt_forge_models/` which is
out of scope for this skill.

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach compilation

## Files changed
- tests/benchmark/test_llms.py (test function added, but model fails to load)

## tt-forge-models submodule
no change
