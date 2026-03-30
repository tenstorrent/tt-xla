# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ChenkinNoob-XL Rectified Flow (ChenkinRF/ChenkinNoob-XL-v0.3-Rectified-Flow) model loader.

ChenkinNoob-XL is a Rectified Flow variant of Stable Diffusion XL, fine-tuned on
Danbooru for anime/illustration-style text-to-image generation. It uses SD3-style
flow-matching sampling rather than standard epsilon/v-prediction scheduling.

Available variants:
- V0_3: ChenkinNoob-XL v0.3 Rectified Flow
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

REPO_ID = "ChenkinRF/ChenkinNoob-XL-v0.3-Rectified-Flow"
CHECKPOINT_FILE = "ChenkinNoob-XL-v0.3-Rectified-Flow.safetensors"


class ModelVariant(StrEnum):
    """Available ChenkinNoob-XL Rectified Flow model variants."""

    V0_3 = "v0.3"


class ModelLoader(ForgeModel):
    """ChenkinNoob-XL Rectified Flow model loader implementation."""

    _VARIANTS = {
        ModelVariant.V0_3: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.V0_3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ChenkinNoob-XL Rectified Flow",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ChenkinNoob-XL pipeline from single-file checkpoint.

        Returns:
            StableDiffusionXLPipeline: The loaded pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        model_path = hf_hub_download(repo_id=REPO_ID, filename=CHECKPOINT_FILE)
        self.pipeline = StableDiffusionXLPipeline.from_single_file(
            model_path,
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
            "masterpiece, best quality, aesthetic, 1girl, solo, looking at viewer"
        ] * batch_size
