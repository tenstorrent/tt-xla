transformers uplift: model-test-uplifts — no uplift-induced failures

All 39 failures in the current context already appear in baseline_failures.txt
with the same root cause, so they were broken on main before the 5.5.1 -> 5.5.2
uplift. None are transformers-API regressions and none are fixable at the
source level without violating the skill's Rules (no PCC-threshold drops for
pre-existing fails, no infra/kernel-build workarounds). Left for human review.

## Skipped (left for human review)
- gemma 1.1_7B_IT / 1.1_2B_IT: pre-existing on baseline (same traceback) — not uplift-induced
- gpt_oss 20B tensor_parallel: pre-existing PCC fail on baseline (~-0.017 vs req 0.98), run-to-run noise — not uplift-induced
- maskformer_swin_b Swin_Base_Coco: pre-existing on baseline (same root cause) — not uplift-induced
- mistral Large_INSTRUCT_2411 / Ministral_8B_Instruct: pre-existing on baseline (same root cause) — not uplift-induced
- olmo3 3_1125_32b / 3_32b_think: pre-existing PCC fail on baseline (same values) — not uplift-induced
- phi4 Phi_4: pre-existing PCC fail on baseline (same value) — not uplift-induced
- qwen_2_5 72B / 72B_Instruct / 7B_Instruct (tensor_parallel + decode): pre-existing on baseline (same root cause) — not uplift-induced
- qwen_2_5 7B_Instruct llm_decode: current run is a tilize_wh jit-build/link flake (kernel cache), baseline was PCC — neither transformers-related nor source-fixable
- transfuser single_device: pre-existing on baseline (same root cause) — not uplift-induced
- yolos_small Small training: pre-existing on baseline (same root cause) — not uplift-induced
- llama causal_lm 3.1_70B / 3.1_70B_Instruct / 3.3_70B_Instruct (tensor_parallel/decode): pre-existing TT_FATAL topology "Failed to add pinning constraints" on baseline — infra, not uplift-induced
- llama causal_lm 3.1_8B_Instruct (all fsdp/tensor_parallel prefill variants): pre-existing on baseline (same root cause) — not uplift-induced
- llama causal_lm 3.2_1B + llama_lora 3.2_1B (prefill training): pre-existing on baseline (same root cause) — not uplift-induced
- phi1 / phi1_lora Phi_1 (prefill training): pre-existing Buffer::allocate_impl on baseline — infra, not uplift-induced

## Stats
- Failures input: 39
- Fixed: 0
- Skipped: 39
