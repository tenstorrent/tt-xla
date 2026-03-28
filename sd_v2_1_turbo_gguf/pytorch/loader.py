# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion v2.1 Turbo GGUF model loader implementation.

Loads GGUF-quantized UNet from gpustack/stable-diffusion-v2-1-turbo-GGUF
and builds a text-to-image pipeline using the base stabilityai/sd-turbo model.

SD-Turbo is a distilled version of Stable Diffusion 2.1, trained using
Adversarial Diffusion Distillation (ADD) for high-quality single-step
image generation at 512x512 resolution.

Available variants:
- Q4_0: 4-bit quantization (~2.19 GB)
- Q4_1: 4-bit quantization (~2.2 GB)
- Q8_0: 8-bit quantization (~2.32 GB)
"""

from typing import Optional

import torch

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

GGUF_REPO = "gpustack/stable-diffusion-v2-1-turbo-GGUF"
BASE_PIPELINE = "stabilityai/sd-turbo"


class ModelVariant(StrEnum):
    """Available Stable Diffusion v2.1 Turbo GGUF variants."""

    Q4_0 = "Q4_0"
    Q4_1 = "Q4_1"
    Q8_0 = "Q8_0"


_GGUF_FILES = {
    ModelVariant.Q4_0: "stable-diffusion-v2-1-turbo-Q4_0.gguf",
    ModelVariant.Q4_1: "stable-diffusion-v2-1-turbo-Q4_1.gguf",
    ModelVariant.Q8_0: "stable-diffusion-v2-1-turbo-Q8_0.gguf",
}


class ModelLoader(ForgeModel):
    """Stable Diffusion v2.1 Turbo GGUF model loader."""

    _VARIANTS = {
        ModelVariant.Q4_0: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
        ModelVariant.Q4_1: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
        ModelVariant.Q8_0: ModelConfig(
            pretrained_model_name=GGUF_REPO,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.Q8_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SD_V2_1_TURBO_GGUF",
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
        """Load the GGUF-quantized UNet and build the SD-Turbo pipeline.

        Uses diffusers GGUFQuantizationConfig to load the quantized UNet,
        then constructs the StableDiffusionPipeline with the base model's
        other components.
        """
        from diffusers import (
            GGUFQuantizationConfig,
            StableDiffusionPipeline,
            UNet2DConditionModel,
        )

        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_file = _GGUF_FILES[self._variant]
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        unet = UNet2DConditionModel.from_single_file(
            f"https://huggingface.co/{GGUF_REPO}/resolve/main/{gguf_file}",
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
        )

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            BASE_PIPELINE,
            unet=unet,
            torch_dtype=compute_dtype,
        )

        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for SD-Turbo.

        Returns:
            list: A list of sample text prompts.
        """
        prompt = [
            "A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
        ] * batch_size
        return prompt
