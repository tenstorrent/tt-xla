# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Boreal-Qwen-Image model loader implementation for text-to-image generation.

Loads LoRA adapter weights from kudzueye/boreal-qwen-image on top of the
Qwen/Qwen-Image base diffusion model. Four style-specific LoRA variants
are available.

Available variants:
- BLEND_LOW_RANK: Boreal blend style (low rank)
- GENERAL_DISCRETE_LOW_RANK: General discrete style (low rank)
- PORTRAITS_HIGH_RANK: Portrait style (high rank)
- SMALL_DISCRETE_LOW_RANK: Small discrete style (low rank)
"""

from typing import Any, Dict, Optional

import torch
from diffusers import DiffusionPipeline

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

LORA_REPO_ID = "kudzueye/boreal-qwen-image"
BASE_MODEL_ID = "Qwen/Qwen-Image"


class ModelVariant(StrEnum):
    """Available Boreal-Qwen-Image LoRA variants."""

    BLEND_LOW_RANK = "blend-low-rank"
    GENERAL_DISCRETE_LOW_RANK = "general-discrete-low-rank"
    PORTRAITS_HIGH_RANK = "portraits-high-rank"
    SMALL_DISCRETE_LOW_RANK = "small-discrete-low-rank"


_LORA_FILES = {
    ModelVariant.BLEND_LOW_RANK: "qwen-boreal-blend-low-rank.safetensors",
    ModelVariant.GENERAL_DISCRETE_LOW_RANK: "qwen-boreal-general-discrete-low-rank.safetensors",
    ModelVariant.PORTRAITS_HIGH_RANK: "qwen-boreal-portraits-portraits-high-rank.safetensors",
    ModelVariant.SMALL_DISCRETE_LOW_RANK: "qwen-boreal-small-discrete-low-rank.safetensors",
}


class ModelLoader(ForgeModel):
    """Boreal-Qwen-Image LoRA model loader for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.BLEND_LOW_RANK: ModelConfig(
            pretrained_model_name=LORA_REPO_ID,
        ),
        ModelVariant.GENERAL_DISCRETE_LOW_RANK: ModelConfig(
            pretrained_model_name=LORA_REPO_ID,
        ),
        ModelVariant.PORTRAITS_HIGH_RANK: ModelConfig(
            pretrained_model_name=LORA_REPO_ID,
        ),
        ModelVariant.SMALL_DISCRETE_LOW_RANK: ModelConfig(
            pretrained_model_name=LORA_REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GENERAL_DISCRETE_LOW_RANK

    DEFAULT_PROMPT = "photo of a serene boreal forest landscape at golden hour"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Boreal-Qwen-Image",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(
        self,
        dtype_override: Optional[torch.dtype] = None,
    ) -> DiffusionPipeline:
        """Load the base Qwen-Image pipeline and apply LoRA weights."""
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = DiffusionPipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=dtype,
        )

        self.pipeline.load_lora_weights(
            LORA_REPO_ID,
            weight_name=_LORA_FILES[self._variant],
        )

        return self.pipeline

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load and return the Boreal-Qwen-Image pipeline with LoRA weights."""
        if self.pipeline is None:
            return self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype=dtype_override)

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Prepare text-to-image generation inputs."""
        prompt_value = prompt if prompt is not None else self.DEFAULT_PROMPT
        return {"prompt": prompt_value}
