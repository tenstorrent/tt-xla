# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MLbackup/5_2025 model loader implementation.

This is a collection of merged Stable Diffusion XL checkpoints based on
Illustrious-XL for text-to-image generation.

Available variants:
- DAWNTREADER_ILLSXL: Dawntreader-ILLSXL merged SDXL checkpoint
"""

from typing import Optional

import torch
from diffusers import StableDiffusionXLPipeline
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


REPO_ID = "MLbackup/5_2025"
SAFETENSORS_FILE = "Dawntreader-ILLSXL.fp16.safetensors"


class ModelVariant(StrEnum):
    """Available MLbackup 5_2025 model variants."""

    DAWNTREADER_ILLSXL = "Dawntreader_ILLSXL"


class ModelLoader(ForgeModel):
    """MLbackup 5_2025 model loader implementation."""

    _VARIANTS = {
        ModelVariant.DAWNTREADER_ILLSXL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.DAWNTREADER_ILLSXL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MLbackup_5_2025",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SDXL pipeline from a single-file checkpoint.

        Returns:
            StableDiffusionXLPipeline: The pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        ckpt_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=SAFETENSORS_FILE,
        )
        self.pipeline = StableDiffusionXLPipeline.from_single_file(
            ckpt_path,
            torch_dtype=dtype,
            **kwargs,
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
