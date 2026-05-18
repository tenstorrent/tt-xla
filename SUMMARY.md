loader_path: third_party.tt_forge_models.beetlelm.causal_lm.pytorch.loader
variant_id: beetlelm_deu_L1_eng_L2_balanced
arch: p150
status: DONE_FAIL
test_function: test_beetlelm_deu_l1_eng_l2_balanced
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
failure_reason: "model weights not found: BeetleLM/beetlelm_deu_L1-eng_L2_balanced does not appear to have a file named pytorch_model.bin or model.safetensors"

# Benchmark added: test_beetlelm_deu_l1_eng_l2_balanced

## Test
tests/benchmark/test_llms.py::test_beetlelm_deu_l1_eng_l2_balanced

## Model
- HF name:    BeetleLM/beetlelm_deu_L1-eng_L2_balanced
- Loader:     third_party.tt_forge_models.beetlelm.causal_lm.pytorch.loader
- Variant:    ModelVariant.BEETLELM_DEU_L1_ENG_L2_BALANCED

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

## Failure Details
The test failed during model loading with:

    OSError: BeetleLM/beetlelm_deu_L1-eng_L2_balanced does not appear to have
    a file named pytorch_model.bin or model.safetensors.

The HuggingFace repository `BeetleLM/beetlelm_deu_L1-eng_L2_balanced` does not
contain the standard PyTorch weight files expected by `AutoModelForCausalLM.from_pretrained`.
This is a model availability/format issue in the loader, not something fixable
within this skill. No changes to files under `third_party/tt_forge_models/` are
permitted. The fix belongs in the tt-forge-models repo (update the loader to
handle the repo's actual file format, or fix/re-upload the model weights on HF).

## Decode roofline (first decode graph, single-chip)
N/A — test did not reach compilation or execution.

## Files changed
- tests/benchmark/test_llms.py (test function added)

## tt-forge-models submodule
no change
