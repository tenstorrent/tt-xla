#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Move vLLM-unsupported models from generative model_configs.yaml into
unsupported_model_configs.yaml. Uses HuggingFace config.architectures and a
list of vLLM-supported architectures (from vLLM validation error) to decide.
Models whose architecture is not in the supported set are moved to unsupported.

Usage (from repo root, with venv activated):
  source venv/bin/activate
  python scripts/move_unsupported_vllm_models.py [--dry-run]
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

from ruamel.yaml import YAML

# vLLM-supported model architectures (from vLLM ModelConfig validation).
# If a model's config.architectures is empty or none of its entries are in this set, it is moved to unsupported.
VLLM_SUPPORTED_ARCHITECTURES = frozenset(
    {
        "AquilaModel",
        "AquilaForCausalLM",
        "ArceeForCausalLM",
        "ArcticForCausalLM",
        "MiniMaxForCausalLM",
        "MiniMaxText01ForCausalLM",
        "MiniMaxM1ForCausalLM",
        "BaiChuanForCausalLM",
        "BaichuanForCausalLM",
        "BailingMoeForCausalLM",
        "BambaForCausalLM",
        "BloomForCausalLM",
        "ChatGLMModel",
        "ChatGLMForConditionalGeneration",
        "CohereForCausalLM",
        "Cohere2ForCausalLM",
        "DbrxForCausalLM",
        "DeciLMForCausalLM",
        "DeepseekForCausalLM",
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "Dots1ForCausalLM",
        "Ernie4_5ForCausalLM",
        "Ernie4_5_MoeForCausalLM",
        "ExaoneForCausalLM",
        "Exaone4ForCausalLM",
        "FalconForCausalLM",
        "Fairseq2LlamaForCausalLM",
        "GemmaForCausalLM",
        "Gemma2ForCausalLM",
        "Gemma3ForCausalLM",
        "Gemma3nForCausalLM",
        "GlmForCausalLM",
        "Glm4ForCausalLM",
        "Glm4MoeForCausalLM",
        "GptOssForCausalLM",
        "GPT2LMHeadModel",
        "GPTBigCodeForCausalLM",
        "GPTJForCausalLM",
        "GPTNeoXForCausalLM",
        "GraniteForCausalLM",
        "GraniteMoeForCausalLM",
        "GraniteMoeHybridForCausalLM",
        "GraniteMoeSharedForCausalLM",
        "GritLM",
        "Grok1ModelForCausalLM",
        "HunYuanMoEV1ForCausalLM",
        "HunYuanDenseV1ForCausalLM",
        "HCXVisionForCausalLM",
        "InternLMForCausalLM",
        "InternLM2ForCausalLM",
        "InternLM2VEForCausalLM",
        "InternLM3ForCausalLM",
        "JAISLMHeadModel",
        "JambaForCausalLM",
        "LlamaForCausalLM",
        "Llama4ForCausalLM",
        "LLaMAForCausalLM",
        "MambaForCausalLM",
        "FalconMambaForCausalLM",
        "FalconH1ForCausalLM",
        "Mamba2ForCausalLM",
        "MiniCPMForCausalLM",
        "MiniCPM3ForCausalLM",
        "MistralForCausalLM",
        "MixtralForCausalLM",
        "QuantMixtralForCausalLM",
        "MptForCausalLM",
        "MPTForCausalLM",
        "MiMoForCausalLM",
        "NemotronForCausalLM",
        "NemotronHForCausalLM",
        "OlmoForCausalLM",
        "Olmo2ForCausalLM",
        "OlmoeForCausalLM",
        "OPTForCausalLM",
        "OrionForCausalLM",
        "PersimmonForCausalLM",
        "PhiForCausalLM",
        "Phi3ForCausalLM",
        "PhiMoEForCausalLM",
        "Phi4FlashForCausalLM",
        "Plamo2ForCausalLM",
        "QWenLMHeadModel",
        "Qwen2ForCausalLM",
        "Qwen2MoeForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3MoeForCausalLM",
        "RWForCausalLM",
        "Step3TextForCausalLM",
        "StableLMEpochForCausalLM",
        "StableLmForCausalLM",
        "Starcoder2ForCausalLM",
        "SolarForCausalLM",
        "TeleChat2ForCausalLM",
        "TeleFLMForCausalLM",
        "XverseForCausalLM",
        "Zamba2ForCausalLM",
        "BartModel",
        "BartForConditionalGeneration",
        "MBartForConditionalGeneration",
        "BertModel",
        "Gemma2Model",
        "GPT2ForSequenceClassification",
        "GteModel",
        "GteNewModel",
        "InternLM2ForRewardModel",
        "JambaForSequenceClassification",
        "LlamaModel",
        "MistralModel",
        "ModernBertModel",
        "NomicBertModel",
        "Qwen2Model",
        "Qwen2ForRewardModel",
        "Qwen2ForProcessRewardModel",
        "RobertaForMaskedLM",
        "RobertaModel",
        "XLMRobertaModel",
        "LlavaNextForConditionalGeneration",
        "Phi3VForCausalLM",
        "Qwen2VLForConditionalGeneration",
        "PrithviGeoSpatialMAE",
        "BertForSequenceClassification",
        "RobertaForSequenceClassification",
        "XLMRobertaForSequenceClassification",
        "ModernBertForSequenceClassification",
        "JinaVLForRanking",
        "AriaForConditionalGeneration",
        "AyaVisionForConditionalGeneration",
        "Blip2ForConditionalGeneration",
        "ChameleonForConditionalGeneration",
        "Cohere2VisionForConditionalGeneration",
        "DeepseekVLV2ForCausalLM",
        "FuyuForCausalLM",
        "Gemma3ForConditionalGeneration",
        "Gemma3nForConditionalGeneration",
        "GLM4VForCausalLM",
        "Glm4vForConditionalGeneration",
        "Glm4vMoeForConditionalGeneration",
        "GraniteSpeechForConditionalGeneration",
        "H2OVLChatModel",
        "InternVLChatModel",
        "InternS1ForConditionalGeneration",
        "Idefics3ForConditionalGeneration",
        "SmolVLMForConditionalGeneration",
        "KeyeForConditionalGeneration",
        "KimiVLForConditionalGeneration",
        "Llama_Nemotron_Nano_VL",
        "LlavaForConditionalGeneration",
        "LlavaNextVideoForConditionalGeneration",
        "LlavaOnevisionForConditionalGeneration",
        "MantisForConditionalGeneration",
        "MiniMaxVL01ForConditionalGeneration",
        "MiniCPMO",
        "MiniCPMV",
        "Mistral3ForConditionalGeneration",
        "MolmoForCausalLM",
        "NVLM_D",
        "Ovis",
        "PaliGemmaForConditionalGeneration",
        "Phi4MMForCausalLM",
        "Phi4MultimodalForCausalLM",
        "PixtralForConditionalGeneration",
        "QwenVLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen2AudioForConditionalGeneration",
        "Qwen2_5OmniModel",
        "Qwen2_5OmniForConditionalGeneration",
        "UltravoxModel",
        "Step3VLForConditionalGeneration",
        "TarsierForConditionalGeneration",
        "Tarsier2ForConditionalGeneration",
        "VoxtralForConditionalGeneration",
        "Florence2ForConditionalGeneration",
        "MllamaForConditionalGeneration",
        "Llama4ForConditionalGeneration",
        "SkyworkR1VChatModel",
        "WhisperForConditionalGeneration",
        "MiMoMTPModel",
        "EagleLlamaForCausalLM",
        "EagleLlama4ForCausalLM",
        "EagleMiniCPMForCausalLM",
        "Eagle3LlamaForCausalLM",
        "DeepSeekMTPModel",
        "Glm4MoeMTPModel",
        "MedusaModel",
        "SmolLM3ForCausalLM",
        "Emu3ForConditionalGeneration",
        "TransformersModel",
        "TransformersForCausalLM",
        "TransformersForMultimodalLM",
    }
)


def _get_model_architecture(model_id: str) -> str | None:
    """Return the first architecture from HF config, or None if unavailable."""
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        archs = getattr(config, "architectures", None) or []
        return archs[0] if archs else None
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Move vLLM-unsupported models from model_configs.yaml to unsupported_model_configs.yaml."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be moved without writing.",
    )
    args = parser.parse_args()

    config_path = (
        PROJECT_ROOT
        / "tests/integrations/vllm_plugin/generative/test_config/model_configs.yaml"
    )
    unsupported_path = (
        PROJECT_ROOT
        / "tests/integrations/vllm_plugin/generative/test_config/unsupported_model_configs.yaml"
    )

    if not config_path.exists():
        print(f"Error: not found {config_path}", file=sys.stderr)
        return 1

    yaml = YAML(typ="rt")
    yaml.preserve_quotes = True
    yaml.width = 4096

    with open(config_path, "r") as f:
        data = yaml.load(f) or {}
    model_configs = data.get("model_configs", data)
    if not hasattr(model_configs, "items"):
        print("Error: no model_configs mapping in YAML", file=sys.stderr)
        return 1
    model_configs = dict(model_configs)

    supported = {}
    unsupported = {}
    unknown_arch = []

    for key, entry in model_configs.items():
        model_id = (entry.get("model") or "").strip()
        if not model_id:
            supported[key] = entry
            continue
        arch = _get_model_architecture(model_id)
        if arch is None:
            # Keep in main config if we can't determine (e.g. offline)
            supported[key] = entry
            unknown_arch.append((key, model_id))
            continue
        if arch in VLLM_SUPPORTED_ARCHITECTURES:
            supported[key] = entry
        else:
            unsupported[key] = entry
            print(
                f"  Unsupported: {key} -> {model_id} (architecture: {arch})",
                file=sys.stderr,
            )

    print(
        f"Supported: {len(supported)}, Unsupported: {len(unsupported)}, Unknown arch: {len(unknown_arch)}",
        file=sys.stderr,
    )

    if args.dry_run:
        print("\nWould move to unsupported_model_configs.yaml:")
        for k in sorted(unsupported.keys()):
            print(f"  {k}: {unsupported[k].get('model')}")
        return 0

    if not unsupported:
        print("No unsupported models to move.")
        return 0

    # Write back model_configs.yaml with only supported
    data["model_configs"] = supported
    with open(config_path, "w") as f:
        yaml.dump(data, f)
    print(f"Wrote {len(supported)} supported entries -> {config_path}")

    # Write unsupported_model_configs.yaml
    header = """# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# vLLM generative models that are not supported by vLLM (architecture not in vLLM supported list).
# Moved by scripts/move_unsupported_vllm_models.py. Do not run these in test_generative_models.

"""
    with open(unsupported_path, "w") as f:
        f.write(header)
        yaml.dump({"model_configs": unsupported}, f)
    print(f"Wrote {len(unsupported)} unsupported entries -> {unsupported_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
