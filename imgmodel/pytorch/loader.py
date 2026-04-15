# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
KoboldCpp imgmodel GGUF model loader implementation for text-to-image generation.

Loads the GGUF-quantized UNet from koboldcpp/imgmodel and builds a
text-to-image pipeline using the base Stable Diffusion v1.5 model.
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

REPO_ID = "koboldcpp/imgmodel"
BASE_PIPELINE = "stable-diffusion-v1-5/stable-diffusion-v1-5"


class ModelVariant(StrEnum):
    """Available koboldcpp imgmodel variants."""

    FTUNED_Q4_0 = "Ftuned_Q4_0"


_GGUF_FILES = {
    ModelVariant.FTUNED_Q4_0: "imgmodel_ftuned_q4_0.gguf",
}


class ModelLoader(ForgeModel):
    """KoboldCpp imgmodel GGUF model loader for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.FTUNED_Q4_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FTUNED_Q4_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="KoboldCpp_ImgModel",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the GGUF-quantized UNet and build the SD pipeline.

        Uses diffusers GGUFQuantizationConfig to load the quantized UNet,
        then constructs the StableDiffusionPipeline with the base model's
        other components (CLIP text encoder, VAE, scheduler).
        """
        from diffusers import (
            GGUFQuantizationConfig,
            StableDiffusionPipeline,
            UNet2DConditionModel,
        )
        from huggingface_hub import hf_hub_download

        dtype = dtype_override if dtype_override is not None else torch.float32
        gguf_filename = _GGUF_FILES[self._variant]
        gguf_path = hf_hub_download(REPO_ID, filename=gguf_filename)

        quantization_config = GGUFQuantizationConfig(compute_dtype=dtype)
        unet = UNet2DConditionModel.from_single_file(
            gguf_path,
            quantization_config=quantization_config,
            torch_dtype=dtype,
        )

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            BASE_PIPELINE,
            unet=unet,
            torch_dtype=dtype,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample text prompts for the model."""
        return ["A cinematic photo of an astronaut riding a horse on mars"] * batch_size
