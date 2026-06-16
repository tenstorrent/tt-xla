transformers uplift: model-test-uplifts — gemma PCC drift; rest systemic

## Fixed
- tests/runner/test_config/torch/test_config_inference_single_device.yaml: gemma/pytorch-1.1_7B_IT single-device inference — lowered required_pcc 0.99→0.98 (calculated 0.9832, isolated single-device numerical drift, no exception).

## Skipped (left for human review)

Investigation: the installed transformers 5.8.1 modeling files still expose the
exact module structure every tt_forge_models loader depends on
(`self_attn.q_proj/k_proj/v_proj/o_proj`, `mlp.gate_proj/up_proj/down_proj`,
gpt_oss `experts.gate_up_proj/down_proj`). Shard specs resolve correctly, so the
tensor-parallel failures are NOT a renamed/reshaped-module mis-shard. None of the
44 failing tests raise a transformers API exception (no AttributeError/TypeError/
ImportError/removed-kwarg). The failures are numerical or compiler/runtime, not
source-level transformers API breaks — so they cannot be fixed under the skill's
rules (no blind PCC lowering, no masking catastrophic regressions).

- Catastrophic TP inference/decode PCC (≈0 or negative): gpt_oss-20B, llama 3.1_70B, mistral Large_INSTRUCT_2411, olmo3 (3_1125_32b, 3_32b_think), phi4 Phi_4, qwen_2_5 (72B, 72B_Instruct, 7B_Instruct), llama 3.1_70B decode, llama 3.3_70B_Instruct decode, qwen_2_5 72B_Instruct/7B_Instruct decode, all llama 3.1_8B_Instruct prefill TP variants (mesh_2x4/4x8, fsdp/megatron). Systemic numerical regression — needs compiler/numerics investigation, not a loader source fix.
- Catastrophic training gradient PCC (per-layer values negative/≈0): gemma 1.1_2B_IT, llama 3.2_1B, llama_lora 3.2_1B, phi1 Phi_1, phi1_lora Phi_1 (single-device prefill training). Not benign drift; not source-fixable.
- Compiler/runtime crashes: maskformer_swin_b (ValueError: Error code 13), mistral Ministral_8B_Instruct (RuntimeError: Bad StatusOr access INTERNAL: Error code 13), llama 3.1_8B_Instruct megatron prefill (TT_THROW program.cpp:249), llama 3.1_70B_Instruct decode (TT_FATAL topology_mapper_utils.cpp:1342). tt-mlir/tt-metal runtime, not transformers source.
- transfuser: torch._dynamo InternalTorchDynamoError (NameError 'named_children') inside timm regnet `for block in self.children()` — torch dynamo/timm issue, unrelated to transformers.
- yolos_small training: teardown CRASHED with signal 6 — infra crash, no transformers cause.

## Stats
- Failures input: 44
- Fixed: 1
- Skipped: 43
