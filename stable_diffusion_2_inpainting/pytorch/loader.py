# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion 2 Inpainting model loader implementation
"""

import torch
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
from diffusers import StableDiffusionInpaintPipeline
from typing import Optional


class ModelVariant(StrEnum):
    """Available Stable Diffusion 2 Inpainting model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Stable Diffusion 2 Inpainting model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="sd2-community/stable-diffusion-2-inpainting",
        )
    }

    DEFAULT_VARIANT = ModelVariant.BASE

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
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """

        return ModelInfo(
            model="Stable Diffusion 2 Inpainting",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Stable Diffusion 2 Inpainting pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use torch.bfloat16.

        Returns:
            StableDiffusionInpaintPipeline: The pre-trained inpainting pipeline object.
        """
        dtype = dtype_override or torch.bfloat16
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, torch_dtype=dtype, **kwargs
        )
        return pipe

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Stable Diffusion 2 Inpainting model.

        Args:
            dtype_override: This parameter is ignored for this model.
            batch_size: Optional batch size for the inputs.

        Returns:
            dict: Dictionary containing prompt, image, and mask_image inputs.
        """
        from PIL import Image
        import numpy as np

        # Create a sample 512x512 image (solid color with some variation)
        image = Image.fromarray(
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        )

        # Create a sample mask (white region in center to inpaint)
        mask = Image.new("L", (512, 512), 0)
        mask.paste(255, (128, 128, 384, 384))

        prompt = [
            "a photo of an astronaut riding a horse on mars",
        ] * batch_size

        return {
            "prompt": prompt,
            "image": image,
            "mask_image": mask,
        }
