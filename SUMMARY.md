loader_path: third_party.tt_forge_models.ministral_8b_gguf.causal_lm.pytorch.loader
variant_id: 8B_Instruct_2512_GGUF
arch: p150
status: DONE_FAIL
test_function: test_ministral_8b_instruct_2512_gguf
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
failure_reason: "NotImplementedError: Unknown gguf model_type: ministral3 in gguf-py — loader's _patch_transformers_mistral3_gguf monkey-patch is incomplete: changes model_type to ministral3 in load_gguf_checkpoint but get_gguf_hf_weights_map still calls gguf-py which does not recognise ministral3"

# Benchmark added: test_ministral_8b_instruct_2512_gguf

## Test
tests/benchmark/test_llms.py::test_ministral_8b_instruct_2512_gguf

## Model
- HF name:    unsloth/Ministral-3-8B-Instruct-2512-GGUF
- Loader:     third_party.tt_forge_models.ministral_8b_gguf.causal_lm.pytorch.loader
- Variant:    ModelVariant.MINISTRAL_8B_INSTRUCT_2512_GGUF ("8B_Instruct_2512_GGUF")

## Test config landed
- optimization_level:        2
- trace_enabled:             true
- experimental_weight_dtype: "bfp_bf8" (default)
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

## Failure details

The test fails immediately when loading model weights (before any compilation
or device execution). The loader's monkey-patch
`_patch_transformers_mistral3_gguf` is incomplete:

1. It patches `load_gguf_checkpoint` to rename `model_type` from `mistral3` to
   `ministral3` so that transformers' AutoConfig can resolve the model class.
2. However, `AutoModelForCausalLM.from_pretrained` subsequently calls
   `get_gguf_hf_weights_map` in `transformers/modeling_gguf_pytorch_utils.py`,
   which invokes `gguf-py` (the upstream llama.cpp Python library) to read
   the GGUF architecture metadata. The installed `gguf-py` package does not
   recognise `ministral3` as a known architecture, raising:

   ```
   NotImplementedError: Unknown gguf model_type: ministral3 in gguf-py.
   This might because you're using an outdated version of gguf-py package,
   you can install `gguf` package from source refer to
   https://github.com/ggerganov/llama.cpp/tree/master/gguf-py#development
   ```

The fix belongs in the loader (or in the gguf-py package version pinned
in the environment) — editing `third_party/tt_forge_models/` is out of
scope for this skill. The submodule is at `67d8d17e19` (2026-05-14).

Remediation options for the tt-forge-models maintainers:
- Extend the monkey-patch to also register `ministral3` with gguf-py's
  architecture registry before calling `get_gguf_hf_weights_map`.
- Or pin `gguf>=0.10.0` (which includes mistral3/ministral3 support) in
  the environment requirements.

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach compilation step.

## Files changed
- tests/benchmark/test_llms.py (test function added)

## tt-forge-models submodule
no change — submodule at 67d8d17e19 (2026-05-14)
