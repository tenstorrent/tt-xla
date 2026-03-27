# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SDXL-Lightning (ByteDance/SDXL-Lightning) model loader implementation.

SDXL-Lightning is a lightning-fast text-to-image generation model distilled from
Stable Diffusion XL using progressive adversarial diffusion distillation.
It can generate high-quality 1024px images in a few steps.

Available variants:
- SDXL_LIGHTNING_4STEP: ByteDance/SDXL-Lightning 4-step UNet variant
"""

from typing import Optional

import torch
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

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


BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
REPO_ID = "ByteDance/SDXL-Lightning"


class ModelVariant(StrEnum):
    """Available SDXL-Lightning model variants."""

    SDXL_LIGHTNING_4STEP = "SDXL_Lightning_4step"


class ModelLoader(ForgeModel):
    """SDXL-Lightning model loader implementation."""

    _VARIANTS = {
        ModelVariant.SDXL_LIGHTNING_4STEP: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.SDXL_LIGHTNING_4STEP

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SDXL_Lightning",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SDXL-Lightning pipeline.

        Loads the base SDXL pipeline and replaces its UNet with the
        SDXL-Lightning 4-step distilled UNet checkpoint.

        Returns:
            StableDiffusionXLPipeline: The SDXL-Lightning pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        ckpt = "sdxl_lightning_4step_unet.safetensors"

        unet = UNet2DConditionModel.from_config(BASE_MODEL, subfolder="unet").to(dtype)
        unet.load_state_dict(load_file(hf_hub_download(REPO_ID, ckpt)))

        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            BASE_MODEL,
            unet=unet,
            torch_dtype=dtype,
            **kwargs,
        )
        self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
            self.pipeline.scheduler.config, timestep_spacing="trailing"
        )

        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the SDXL-Lightning model.

        Returns:
            list: A list of sample text prompts.
        """
        return ["A girl smiling"] * batch_size
