# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv5 model loader implementation
"""
import torch
from torchvision import transforms
from datasets import load_dataset
from typing import Optional

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


class ModelVariant(StrEnum):
    """Available YOLOv5 model variants."""

    YOLOV5N = "yolov5n"
    YOLOV5S = "yolov5s"
    YOLOV5M = "yolov5m"
    YOLOV5L = "yolov5l"
    YOLOV5X = "yolov5x"


class ModelLoader(ForgeModel):
    """YOLOv5 model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.YOLOV5N: ModelConfig(
            pretrained_model_name="yolov5n",
        ),
        ModelVariant.YOLOV5S: ModelConfig(
            pretrained_model_name="yolov5s",
        ),
        ModelVariant.YOLOV5M: ModelConfig(
            pretrained_model_name="yolov5m",
        ),
        ModelVariant.YOLOV5L: ModelConfig(
            pretrained_model_name="yolov5l",
        ),
        ModelVariant.YOLOV5X: ModelConfig(
            pretrained_model_name="yolov5x",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.YOLOV5S

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

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
            model="yolov5",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.TORCH_HUB,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the YOLOv5 model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The YOLOv5 model instance.
        """
        # Get the model name from the instance's variant config
        model_variant = self._variant_config.pretrained_model_name
        model = torch.hub.load("ultralytics/yolov5", model_variant)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1, input_size=640):
        """Load and return sample inputs for the YOLOv5 model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.
            input_size: Optional input size (width and height) to override the default size of 640.

        Returns:
            torch.Tensor: Sample input tensor that can be fed to the model.
        """

        # Load dataset
        dataset = load_dataset(
            "huggingface/cats-image", split="test"
        )  # cats-image is a dataset of 1000 images of cats

        # Get first image from dataset
        image = dataset[0]["image"]

        # Preprocess the image
        transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
            ]
        )

        img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Replicate tensors for batch size
        batch_tensor = img_tensor.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return batch_tensor
