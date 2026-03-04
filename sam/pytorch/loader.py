# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SAM (Segment Anything Model) loader implementation
"""

import torch
from typing import Optional
from PIL import Image
from loguru import logger
from transformers import SamModel, SamProcessor

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
from datasets import load_dataset


class ModelVariant(StrEnum):
    """Available SAM model variants."""

    BASE = "Vit_Base"
    LARGE = "Vit_Large"
    HUGE = "Vit_Huge"


class ModelLoader(ForgeModel):
    """SAM model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="facebook/sam-vit-base",
        ),
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="facebook/sam-vit-large",
        ),
        ModelVariant.HUGE: ModelConfig(
            pretrained_model_name="facebook/sam-vit-huge",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SAM",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SAM model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The wrapped SAM model instance.
        """
        # Get the pretrained model name from the instance's variant config
        model_name = self._variant_config.pretrained_model_name

        # Load SAM model from transformers
        framework_model = SamModel.from_pretrained(model_name, **kwargs).to("cpu")

        # Load processor for this variant
        self.processor = SamProcessor.from_pretrained(model_name, **kwargs)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            framework_model = framework_model.to(dtype_override)

        return framework_model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the SAM model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            tuple: (pixel_values, input_points) suitable for SAM.
        """
        # Ensure processor is loaded
        if self.processor is None:
            model_name = self._variant_config.pretrained_model_name
            self.processor = SamProcessor.from_pretrained(model_name)

        # Load image from HuggingFace dataset
        try:
            dataset = load_dataset("huggingface/cats-image")["test"]
            raw_image = dataset[0]["image"].convert("RGB")
        except Exception as e:
            logger.warning(
                f"Failed to load image from dataset. Using random fallback tensor. Reason: {e}"
            )
            raw_image = Image.fromarray(
                (torch.rand(3, 1024, 1024) * 255).byte().permute(1, 2, 0).numpy()
            )

        # Define input points for segmentation
        input_points = [[[450, 600]]]

        # Process inputs using SAM processor
        inputs = self.processor(
            raw_image, input_points=input_points, return_tensors="pt"
        ).to("cpu")

        # Extract pixel values and input points
        pixel_values = inputs["pixel_values"]
        input_points_tensor = inputs["input_points"]

        # Replicate tensors for batch size
        pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)
        input_points_tensor = input_points_tensor.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)
            input_points_tensor = input_points_tensor.to(dtype_override)

        return pixel_values, input_points_tensor
