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
    """Available Vit model variants."""

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
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="vit",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
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
        """Load a Vit model from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The ViT model instance.
        """
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
        """Load and return sample inputs for the ViT model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for ViT.
        """
        # Get the Image
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file)

        # Initialize image processor
        image_processor = AutoImageProcessor.from_pretrained(
            self.model_name, use_fast=True
        )

        # Create processed inputs
        inputs = image_processor(images=image, return_tensors="pt").pixel_values

        # Create batch (default 1)
        inputs = inputs.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs

    def post_processing(self, co_out):
        """Print classification results.

        Args:
            co_out: Output from the compiled model
        """
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
