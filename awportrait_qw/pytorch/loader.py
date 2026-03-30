# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AWPortrait-QW LoRA model loader implementation.

Loads the Qwen/Qwen-Image base diffusion pipeline and applies the
Shakker-Labs/AWPortrait-QW LoRA weights for enhanced portrait generation
with a focus on Chinese facial features and aesthetics.

Available variants:
- AWPORTRAIT_QW: AWPortrait-QW LoRA on Qwen-Image
"""

from typing import Any, Optional

import torch
from diffusers import DiffusionPipeline

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

BASE_MODEL = "Qwen/Qwen-Image"
LORA_REPO = "Shakker-Labs/AWPortrait-QW"
LORA_WEIGHT_NAME = "AWPortrait-QW_1.0.safetensors"


class ModelVariant(StrEnum):
    """Available AWPortrait-QW model variants."""

    AWPORTRAIT_QW = "AWPortrait_QW"


class ModelLoader(ForgeModel):
    """AWPortrait-QW LoRA model loader."""

    _VARIANTS = {
        ModelVariant.AWPORTRAIT_QW: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.AWPORTRAIT_QW

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[DiffusionPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="AWPORTRAIT_QW",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the Qwen-Image pipeline with AWPortrait-QW LoRA weights.

        Returns:
            DiffusionPipeline with LoRA weights loaded.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=LORA_WEIGHT_NAME,
        )

        return self.pipeline

    def load_inputs(self, **kwargs) -> Any:
        """Prepare inputs for portrait generation.

        Returns:
            dict with prompt and negative_prompt keys.
        """
        prompt = "a professional portrait photo of a woman in a studio setting"
        negative_prompt = (
            "blurry, bad faces, bad hands, worst quality, "
            "low quality, jpeg artifacts"
        )

        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
        }
