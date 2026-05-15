loader_path: third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader
variant_id: qiaw99_7B_Instruct_OpenbookQA_DPO_C_new
arch: n150
status: DONE_FAIL
test_function: test_qiaw99_7b_instruct_openbookqa_dpo_c_new
samples_per_second: null
ttft_ms: null
prefill_pcc: null
first_decode_pcc: null
top_perf_samples_per_sec: null
pct_of_target: null
roofline_bound: null
optimization_level: null
trace_enabled: null
experimental_weight_dtype: null
failure_reason: "variant qiaw99_7B_Instruct_OpenbookQA_DPO_C_new not in current ModelVariant enum of third_party/tt_forge_models/qwen_2_5/causal_lm/pytorch/loader.py at submodule HEAD 93218a34fc; string never appeared in any commit of that file's git history"

# Benchmark added: test_qiaw99_7b_instruct_openbookqa_dpo_c_new

## Test
tests/benchmark/test_llms.py::test_qiaw99_7b_instruct_openbookqa_dpo_c_new

## Model
- HF name:    qiaw99/7B_Instruct_OpenbookQA_DPO_C_new (inferred from variant)
- Loader:     third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader
- Variant:    qiaw99_7B_Instruct_OpenbookQA_DPO_C_new

## Test config landed
- optimization_level:        N/A (early exit — test not added)
- trace_enabled:             N/A
- experimental_weight_dtype: N/A
- batch_size:                N/A
- input_sequence_length:     N/A
- required_pcc:              N/A

## Measured (full model, defaults)
- Sample per second:  N/A
- TTFT (ms):          N/A
- Prefill PCC:        N/A
- First decode PCC:   N/A
- Wall clock:         N/A
- Hardware:           n150 (wormhole_b0, n300 card single chip)

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A

N/A — early exit before test was run.

## Files changed
- SUMMARY.md (this file only — no test code added)

## tt-forge-models submodule
no change

## Failure details

The variant `qiaw99_7B_Instruct_OpenbookQA_DPO_C_new` does not exist in the
`ModelVariant` enum of `third_party/tt_forge_models/qwen_2_5/causal_lm/pytorch/loader.py`
at the current submodule HEAD (`93218a34fc`). A full `git log -p` search of
that file's history found no mention of the string `qiaw99` at any prior
commit — the variant was never present in the submodule.

Current enum members as of submodule HEAD 93218a34fc:
- QWEN_2_5_0_5B, QWEN_2_5_0_5B_INSTRUCT
- QWEN_2_5_1_5B, QWEN_2_5_1_5B_INSTRUCT
- QWEN_2_5_3B, QWEN_2_5_3B_INSTRUCT
- QWEN_2_5_7B, QWEN_2_5_7B_INSTRUCT, QWEN_2_5_7B_INSTRUCT_1M
- QWEN_2_5_14B, QWEN_2_5_14B_INSTRUCT, QWEN_2_5_14B_INSTRUCT_1M
- QWEN_2_5_32B_INSTRUCT, QWEN_2_5_72B_INSTRUCT, QWEN_2_5_72B
- QWEN_2_5_MATH_7B

The remediation_hash provided was `null`, so there is no historical submodule
commit to pin to. The fix must be made in the tt-forge-models repo by adding
this variant to the `ModelVariant` enum and providing the corresponding
`pretrained_model_name` entry. This skill will not modify files under
`third_party/tt_forge_models/`.
