loader_path: third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader
variant_id: QIAW99_QWEN_2_5_7B_INSTRUCT_OPENBOOKQA_DPO_C_NEW
arch: n150
status: DONE_FAIL
test_function: test_qiaw99_qwen_2_5_7b_instruct_openbookqa_dpo_c_new
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
failure_reason: "variant QIAW99_QWEN_2_5_7B_INSTRUCT_OPENBOOKQA_DPO_C_NEW not in current ModelVariant enum of third_party/tt_forge_models/qwen_2_5/causal_lm/pytorch/loader.py at submodule HEAD 93218a34fc; variant did exist in older commits (last seen in 37e14db384: Add unsloth/Qwen2.5-3B causal LM variant)"

# Benchmark added: test_qiaw99_qwen_2_5_7b_instruct_openbookqa_dpo_c_new

## Test
tests/benchmark/test_llms.py::test_qiaw99_qwen_2_5_7b_instruct_openbookqa_dpo_c_new

## Model
- HF name:    qiaw99/Qwen2.5-7B-Instruct-OpenbookQA-DPO-C-new
- Loader:     third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader
- Variant:    QIAW99_QWEN_2_5_7B_INSTRUCT_OPENBOOKQA_DPO_C_NEW ("qiaw99_7B_Instruct_OpenbookQA_DPO_C_new")

## Early Exit Reason

Variant `QIAW99_QWEN_2_5_7B_INSTRUCT_OPENBOOKQA_DPO_C_NEW` is **not present** in the current `ModelVariant` enum at submodule HEAD `93218a34fc`. The variant did exist in older submodule commits (last seen in commit `37e14db384: Add unsloth/Qwen2.5-3B causal LM variant`), but has since been removed from the loader.

Per skill policy, modifying files under `third_party/tt_forge_models/` is out of scope. The fix (restoring or renaming the variant) must be done in the tt-forge-models repo. Once the variant is present at the current (or a newer) submodule HEAD, this benchmark can be re-attempted.

To unblock: pin the submodule to a commit where the variant exists (e.g. `37e14db384`) or re-add the variant in tt-forge-models.

## Test config landed
- optimization_level:        N/A
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
- Hardware:           n150 (Wormhole, n300 board)

## Decode roofline (first decode graph, single-chip)
Source JSON: N/A (test not run)
Achieved vs top_perf_samples_per_sec: N/A

## Files changed
- SUMMARY.md (this file only — no test code added)

## tt-forge-models submodule
no change (submodule remains at 93218a34fc)
