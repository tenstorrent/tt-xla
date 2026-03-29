# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
KoboldCpp imgmodel GGUF model loader implementation for text-to-image generation.

Loads quantized Stable Diffusion models in GGUF format from the
koboldcpp/imgmodel repository on Hugging Face.
"""

from typing import Optional

import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download

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


class ModelVariant(StrEnum):
    """Available koboldcpp imgmodel variants."""

    FTUNED_Q4_0 = "Ftuned_Q4_0"


class ModelLoader(ForgeModel):
    """KoboldCpp imgmodel GGUF model loader for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.FTUNED_Q4_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FTUNED_Q4_0

    _GGUF_FILES = {
        ModelVariant.FTUNED_Q4_0: "imgmodel_ftuned_q4_0.gguf",
    }

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
        """Load the GGUF Stable Diffusion model from koboldcpp/imgmodel."""
        gguf_filename = self._GGUF_FILES[self._variant]
        gguf_path = hf_hub_download(REPO_ID, filename=gguf_filename)

        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionPipeline.from_single_file(
            gguf_path,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample text prompts for the model."""
        return ["A cinematic photo of an astronaut riding a horse on mars"] * batch_size
