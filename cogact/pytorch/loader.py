# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CogACT model loader implementation for action prediction.

CogACT is a Vision-Language-Action model that extends a Prismatic VLM
(DINOv2 + SigLIP + LLaMA-2 7B) with a DiT-based diffusion action head
for robotic manipulation tasks.
"""
import torch
from typing import Optional

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...openvla.pytorch.src.configuration_prismatic import PrismaticConfig
from ...openvla.pytorch.src.modeling_prismatic import PrismaticForConditionalGeneration
from ...openvla.pytorch.src.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)
from .src.model import CogACTWrapper, DiT


class ModelVariant(StrEnum):
    """Available CogACT model variants."""

    COGACT_BASE = "Base"


class ModelLoader(ForgeModel):
    """CogACT model loader implementation for action prediction tasks."""

    _VARIANTS = {
        ModelVariant.COGACT_BASE: ModelConfig(
            pretrained_model_name="CogACT/CogACT-Base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.COGACT_BASE

    sample_prompt = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "
        "USER: What action should the robot take to move the sponge near the apple? ASSISTANT:"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="CogACT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_ACTION_PREDICTION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor using Prismatic components.

        CogACT shares the same Prismatic VLM base as OpenVLA (prism-dinosiglip-224px+7b),
        so we load the processor from the OpenVLA model.
        """
        image_processor = PrismaticImageProcessor.from_pretrained("openvla/openvla-7b")
        tokenizer = AutoTokenizer.from_pretrained("openvla/openvla-7b")
        self.processor = PrismaticProcessor(
            image_processor=image_processor, tokenizer=tokenizer
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the CogACT model instance.

        Creates the Prismatic VLM from config, adds a DiT action model,
        and loads weights from the CogACT checkpoint.
        """
        repo_id = self._variant_config.pretrained_model_name

        # Download checkpoint from HuggingFace
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename="checkpoints/CogACT-Base.pt",
        )

        # Create VLM with the same architecture as CogACT's base
        # (DINOv2+SigLIP at 224px, LLaMA-2 7B)
        vlm_config = PrismaticConfig(
            vision_backbone_id="dinosiglip-vit-so-224px",
            llm_backbone_id="llama2-7b-pure",
            arch_specifier="no-align+gelu-mlp",
        )
        vlm = PrismaticForConditionalGeneration(vlm_config)

        # Create DiT action model (DiT-Base)
        action_model = DiT(
            action_dim=7,
            hidden_size=768,
            depth=12,
            num_heads=12,
            conditioning_dim=vlm_config.text_config.hidden_size,
            num_action_tokens=16,
        )

        # Load weights from CogACT checkpoint (weights_only=False required for
        # raw .pt checkpoint containing nested dicts with custom structure)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "model" in checkpoint:
            model_weights = checkpoint["model"]
            vlm_state_dict = {}
            for key, value in model_weights.items():
                if key.startswith("projector."):
                    vlm_state_dict[key] = value
                elif key.startswith("llm_backbone."):
                    new_key = key.replace("llm_backbone.", "language_model.", 1)
                    vlm_state_dict[new_key] = value
                elif key.startswith("vision_backbone."):
                    vlm_state_dict[key] = value
            vlm.load_state_dict(vlm_state_dict, strict=False)

        if "action_model" in checkpoint:
            action_model.load_state_dict(checkpoint["action_model"], strict=False)

        # Wrap VLM + DiT action model
        model = CogACTWrapper(vlm, action_model)

        if dtype_override is None:
            dtype_override = torch.float32
        model = model.to(dtype_override)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the CogACT model."""
        if self.processor is None:
            self._load_processor()

        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"].convert("RGB")

        inputs = self.processor(self.sample_prompt, image)

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
