# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DINOv2 model loader implementation for image classification.
"""
import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)
from typing import Optional
from PIL import Image

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
from ....tools.utils import get_file


class ModelVariant(StrEnum):
    """Available DINOv2 model variants for image classification."""

    SMALL = "small"
    BASE = "base"
    LARGE = "large"
    GIANT = "giant"


class ModelLoader(ForgeModel):
    """DINOv2 model loader implementation for image classification tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.SMALL: ModelConfig(
            pretrained_model_name="facebook/dinov2-small-imagenet1k-1-layer",
        ),
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="facebook/dinov2-base-imagenet1k-1-layer",
        ),
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="facebook/dinov2-large-imagenet1k-1-layer",
        ),
        ModelVariant.GIANT: ModelConfig(
            pretrained_model_name="facebook/dinov2-giant-imagenet1k-1-layer",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.SMALL

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

        if variant == ModelVariant.SMALL:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="dinov2",
            variant=variant,
            group=group,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load image processor for the current variant.

        Returns:
            The loaded processor instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Load AutoImageProcessor
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)

        return self.processor

    def load_model(self, dtype_override=None):
        """Load and return the DINOv2 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The DINOv2 model instance for image classification.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        # Load the model with dtype override if specified
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AutoModelForImageClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the DINOv2 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure processor is initialized
        if self.processor is None:
            self._load_processor()

        # Get the Image
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file)

        # Process images
        inputs = self.processor(images=image, return_tensors="pt")

        # Handle batch size
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Convert the input dtype to dtype_override if specified
        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].dtype.is_floating_point:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
