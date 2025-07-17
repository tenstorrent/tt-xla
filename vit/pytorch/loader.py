# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Vit model loader implementation for question answering
"""
import torch
from transformers import AutoImageProcessor, ViTForImageClassification
from PIL import Image
from typing import Optional

from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    ModelConfig,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available ALBERT model variants."""

    BASE = "base"
    LARGE = "large"


class ModelLoader(ForgeModel):
    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="google/vit-base-patch16-224",
        ),
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="google/vit-large-patch16-224",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LARGE

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="vit",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.TORCH_HUB,
            framework=Framework.TORCH,
        )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_name = self._variant_config.pretrained_model_name

    def load_model(self, dtype_override=None):
        """Load a Vit model from Hugging Face."""
        model = ViTForImageClassification.from_pretrained(
            self.model_name, return_dict=False
        )
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)
        self.model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Generate sample inputs for Vit models."""
        # Get the Image
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file)
        # Initialize tokenizer
        image_processor = AutoImageProcessor.from_pretrained(
            self.model_name, use_fast=True
        )

        # Create tokenized inputs
        inputs = image_processor(images=image, return_tensors="pt").pixel_values

        # Creat batch (default 1)
        inputs = inputs.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs

    def post_processing(self, co_out):
        logits = co_out[0]
        predicted_class_indices = logits.argmax(-1)

        # Handle both single and batch predictions
        if predicted_class_indices.dim() == 0:  # Single prediction (scalar)
            print(
                "Predicted class:",
                self.model.config.id2label[predicted_class_indices.item()],
            )
        else:  # Batch predictions
            for i, idx in enumerate(predicted_class_indices):
                class_name = self.model.config.id2label[idx.item()]
                print(f"Batch {i}: Predicted class: {class_name}")
