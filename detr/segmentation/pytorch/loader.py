# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DETR model loader implementation for segmentation.
"""
import torch
from transformers import DetrForSegmentation, DetrFeatureExtractor
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
    """Available DETR model variants for segmentation."""

    RESNET_50_PANOPTIC = "resnet_50_panoptic"


class ModelLoader(ForgeModel):
    """DETR model loader implementation for segmentation tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.RESNET_50_PANOPTIC: ModelConfig(
            pretrained_model_name="facebook/detr-resnet-50-panoptic",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.RESNET_50_PANOPTIC

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.feature_extractor = None

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
            model="detr_segmentation",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_SEMANTIC_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_feature_extractor(self, dtype_override=None):
        """Load feature extractor for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the feature extractor's default dtype.

        Returns:
            The loaded feature extractor instance
        """
        # Initialize feature extractor with dtype override if specified
        extractor_kwargs = {}
        if dtype_override is not None:
            extractor_kwargs["torch_dtype"] = dtype_override

        # Load the feature extractor
        self.feature_extractor = DetrFeatureExtractor.from_pretrained(
            self._variant_config.pretrained_model_name, **extractor_kwargs
        )

        return self.feature_extractor

    def load_model(self, dtype_override=None):
        """Load and return the DETR model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The DETR model instance for segmentation.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure feature extractor is loaded
        if self.feature_extractor is None:
            self._load_feature_extractor(dtype_override=dtype_override)

        # Load the model with dtype override if specified
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = DetrForSegmentation.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the DETR model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure feature extractor is initialized
        if self.feature_extractor is None:
            self._load_feature_extractor(dtype_override=dtype_override)

        # Get the Image
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file)
        inputs = self.feature_extractor(images=image, return_tensors="pt")

        # Handle batch size
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
