# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ByteDance Hyper-SD model loader implementation

Hyper-SD provides accelerated diffusion model inference via distilled LoRA
adapters. This loader applies Hyper-SD LoRA weights to the SDXL base model
for fast text-to-image generation with reduced inference steps.
"""

import torch
from typing import Optional

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
from diffusers import DiffusionPipeline, DDIMScheduler
from huggingface_hub import hf_hub_download


LORA_REPO = "ByteDance/Hyper-SD"
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


class ModelVariant(StrEnum):
    """Available Hyper-SD model variants."""

    SDXL_2STEP_LORA = "SDXL_2step_LoRA"
    SDXL_4STEP_LORA = "SDXL_4step_LoRA"
    SDXL_8STEP_LORA = "SDXL_8step_LoRA"


_LORA_FILES = {
    ModelVariant.SDXL_2STEP_LORA: "Hyper-SDXL-2steps-lora.safetensors",
    ModelVariant.SDXL_4STEP_LORA: "Hyper-SDXL-4steps-lora.safetensors",
    ModelVariant.SDXL_8STEP_LORA: "Hyper-SDXL-8steps-lora.safetensors",
}


class ModelLoader(ForgeModel):
    """ByteDance Hyper-SD model loader implementation."""

    _VARIANTS = {
        ModelVariant.SDXL_2STEP_LORA: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.SDXL_4STEP_LORA: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.SDXL_8STEP_LORA: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SDXL_2STEP_LORA

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional variant name string. If None, uses DEFAULT_VARIANT.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="Hyper-SD",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the SDXL pipeline with Hyper-SD LoRA weights applied.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use torch.float16.

        Returns:
            DiffusionPipeline: The SDXL pipeline with Hyper-SD LoRA weights fused.
        """
        dtype = dtype_override or torch.float16
        pipe = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            variant="fp16",
            **kwargs,
        )

        lora_file = _LORA_FILES[self._variant]
        pipe.load_lora_weights(
            hf_hub_download(LORA_REPO, lora_file),
        )
        pipe.fuse_lora()

        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )

        return pipe

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs):
        """Load and return sample text prompts for the Hyper-SD model.

        Args:
            dtype_override: This parameter is ignored for this model.
            batch_size: Optional batch size for the prompts.

        Returns:
            list: A list of sample text prompts.
        """
        prompt = [
            "a photo of an astronaut riding a horse on mars",
        ] * batch_size
        return prompt
