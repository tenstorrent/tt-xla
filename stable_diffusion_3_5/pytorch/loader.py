# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion 3.5 model loader implementation
https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
"""

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
import torch
from diffusers import StableDiffusion3Pipeline, AutoencoderTiny
from typing import Optional


class ModelVariant(StrEnum):
    """Available Stable Diffusion 3.5 model variants."""

    MEDIUM = "medium"
    LARGE = "large"


class ModelLoader(ForgeModel):
    """Stable Diffusion 3.5 model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.MEDIUM: ModelConfig(
            pretrained_model_name="stabilityai/stable-diffusion-3.5-medium",
        ),
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="stabilityai/stable-diffusion-3.5-large",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.MEDIUM

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="stable_diffusion_3_5",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.MM_IMAGE_TTT,  # FIXME: Update to text to image
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the Stable Diffusion 3.5 pipeline for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use torch.bfloat16.

        Returns:
            StableDiffusion3Pipeline: The pre-trained Stable Diffusion 3.5 pipeline object.
        """
        dtype = dtype_override or torch.bfloat16

        pipe = StableDiffusion3Pipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            text_encoder_3=None,
            tokenizer_3=None,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        # Memory optimization recommended by: https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/stable_diffusion_3#tiny-autoencoder-for-stable-diffusion-3
        pipe.vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd3", torch_dtype=dtype, low_cpu_mem_usage=True
        )
        pipe.enable_attention_slicing()
        return pipe

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Stable Diffusion 3.5 model.

        Args:
            dtype_override: This parameter is ignored for this model.
            batch_size: Optional batch size for the inputs.

        Returns:
            dict: Dictionary containing prompt and other generation parameters.
        """
        prompt = [
            "a photo of an astronaut riding a horse on mars",
        ]

        negative_prompt = ""
        height = 512
        width = 512
        guidance_scale = 7.0
        arguments = {
            "prompt": prompt * batch_size,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
        }

        return arguments
