# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLOS Signature Detection model loader implementation for document signature detection.
"""

from transformers import AutoImageProcessor, AutoModelForObjectDetection
from datasets import load_dataset
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ...base import ForgeModel


class ModelLoader(ForgeModel):
    """YOLOS Signature Detection model loader implementation."""

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="YOLOS Signature Detection",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.model_variant = "mdefrance/yolos-base-signature-detection"

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the YOLOS Signature Detection model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The YOLOS model instance.
        """
        variant = self.model_variant
        model = AutoModelForObjectDetection.from_pretrained(variant, **kwargs)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the YOLOS Signature Detection model.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """
        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"]
        image_processor = AutoImageProcessor.from_pretrained(self.model_variant)
        inputs = image_processor(images=image, return_tensors="pt")
        batch_tensor = inputs["pixel_values"]

        batch_tensor = batch_tensor.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return batch_tensor
