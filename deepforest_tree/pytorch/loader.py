# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DeepForest Tree Crown Detection model loader implementation for object detection.
"""
import torch
import numpy as np
from deepforest import main as deepforest_main
from datasets import load_dataset
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


class ModelVariant(StrEnum):
    """Available DeepForest Tree model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """DeepForest Tree Crown Detection model loader implementation for object detection tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="weecology/deepforest-tree",
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
            model="DeepForest-Tree",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DeepForest Tree model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The DeepForest Tree model instance for object detection.
        """
        # Load the DeepForest model and its pretrained weights
        df_model = deepforest_main.deepforest()
        df_model.use_release()

        # Extract the underlying PyTorch detection model
        model = df_model.model

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the DeepForest Tree model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            list: Input tensors as a list of image tensors, as expected by torchvision detection models.
        """
        # Load a sample image from HuggingFace datasets
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"].resize((400, 400))

        # Convert to tensor in [C, H, W] format with values in [0, 1]
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

        if dtype_override is not None:
            image_tensor = image_tensor.to(dtype_override)

        # Torchvision detection models expect a list of image tensors
        inputs = [image_tensor for _ in range(batch_size)]

        return [inputs]
