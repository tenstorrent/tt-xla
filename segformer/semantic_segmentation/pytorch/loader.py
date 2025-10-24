# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Segformer for Semantic Segmentation model loader implementation
"""
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
from ....tools.utils import get_file
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Segformer for Semantic Segmentation model variants."""

    B0_FINETUNED = "b0_finetuned_ade_512_512"
    B1_FINETUNED = "b1_finetuned_ade_512_512"
    B2_FINETUNED = "b2_finetuned_ade_512_512"
    B3_FINETUNED = "b3_finetuned_ade_512_512"
    B4_FINETUNED = "b4_finetuned_ade_512_512"


class ModelLoader(ForgeModel):
    """Segformer model loader implementation for semantic segmentation tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.B0_FINETUNED: ModelConfig(
            pretrained_model_name="nvidia/segformer-b0-finetuned-ade-512-512",
        ),
        ModelVariant.B1_FINETUNED: ModelConfig(
            pretrained_model_name="nvidia/segformer-b1-finetuned-ade-512-512",
        ),
        ModelVariant.B2_FINETUNED: ModelConfig(
            pretrained_model_name="nvidia/segformer-b2-finetuned-ade-512-512",
        ),
        ModelVariant.B3_FINETUNED: ModelConfig(
            pretrained_model_name="nvidia/segformer-b3-finetuned-ade-512-512",
        ),
        ModelVariant.B4_FINETUNED: ModelConfig(
            pretrained_model_name="nvidia/segformer-b4-finetuned-ade-512-512",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.B0_FINETUNED

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
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="segformer_semantic_segmentation",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant.

        Returns:
            The loaded processor instance
        """
        # Initialize processor
        self.processor = SegformerImageProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, dtype_override=None):
        """Load and return the Segformer for Semantic Segmentation model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use bfloat16.

        Returns:
            torch.nn.Module: The Segformer model instance for semantic segmentation modeling.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure processor is loaded
        if self.processor is None:
            self._load_processor()

        # Load pre-trained model from HuggingFace
        model = SegformerForSemanticSegmentation.from_pretrained(pretrained_model_name)

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Segformer for Semantic Segmentation model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors (pixel_values) that can be fed to the model.
        """
        # Ensure processor is initialized
        if self.processor is None:
            self._load_processor()

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(str(image_file))
        inputs = self.processor(images=image, return_tensors="pt")

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        # Add batch dimension
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
