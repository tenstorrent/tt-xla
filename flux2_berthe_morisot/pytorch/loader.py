# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.2 Berthe Morisot style LoRA model loader implementation.

Loads the FLUX.2-dev base pipeline and applies Berthe Morisot style LoRA
weights from ostris/flux2_berthe_morisot for generating images in the
impressionist style of painter Berthe Morisot.

Available variants:
- BERTHE_MORISOT: Berthe Morisot style LoRA applied to FLUX.2-dev
"""

from typing import Any, Optional

import torch
from diffusers import AutoPipelineForText2Image  # type: ignore[import]

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

BASE_MODEL = "black-forest-labs/FLUX.2-dev"
LORA_REPO = "ostris/flux2_berthe_morisot"
LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"


class ModelVariant(StrEnum):
    """Available FLUX.2 Berthe Morisot LoRA variants."""

    BERTHE_MORISOT = "BertheMorisot"


class ModelLoader(ForgeModel):
    """FLUX.2 Berthe Morisot style LoRA model loader."""

    _VARIANTS = {
        ModelVariant.BERTHE_MORISOT: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.BERTHE_MORISOT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX2_BERTHE_MORISOT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the FLUX.2-dev pipeline with Berthe Morisot style LoRA weights.

        Returns:
            Pipeline with LoRA weights loaded.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )

        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=LORA_WEIGHT_NAME,
        )

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for text-to-image generation.

        Returns:
            dict with prompt key.
        """
        if prompt is None:
            prompt = (
                "berthe_morisot style, woman playing the guitar, "
                "on stage, singing a song"
            )

        return {
            "prompt": prompt,
        }
