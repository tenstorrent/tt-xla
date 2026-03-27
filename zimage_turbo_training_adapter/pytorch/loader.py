# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Z-Image Turbo Training Adapter (ostris/zimage_turbo_training_adapter) model loader.

This is a LoRA de-distillation adapter for the Tongyi-MAI/Z-Image-Turbo base model.
It is designed to be stacked during fine-tuning to preserve step-distillation speed.
The adapter loads the base Z-Image-Turbo pipeline and applies the LoRA weights.

Available variants:
- ZIMAGE_TURBO_TRAINING_ADAPTER_V1: ostris/zimage_turbo_training_adapter (v1 weights)
"""

from typing import Optional

import torch
from diffusers import AutoPipelineForText2Image

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


BASE_REPO_ID = "Tongyi-MAI/Z-Image-Turbo"
ADAPTER_REPO_ID = "ostris/zimage_turbo_training_adapter"


class ModelVariant(StrEnum):
    """Available Z-Image Turbo Training Adapter model variants."""

    ZIMAGE_TURBO_TRAINING_ADAPTER_V1 = "ZImage_Turbo_Training_Adapter_v1"


class ModelLoader(ForgeModel):
    """Z-Image Turbo Training Adapter model loader implementation."""

    _VARIANTS = {
        ModelVariant.ZIMAGE_TURBO_TRAINING_ADAPTER_V1: ModelConfig(
            pretrained_model_name=BASE_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.ZIMAGE_TURBO_TRAINING_ADAPTER_V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ZImage_Turbo_Training_Adapter",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Z-Image-Turbo base pipeline and apply the LoRA adapter weights.

        Returns:
            AutoPipelineForText2Image: The pipeline with LoRA adapter applied.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        self.pipeline.load_lora_weights(
            ADAPTER_REPO_ID,
            weight_name="zimage_turbo_training_adapter_v1.safetensors",
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
