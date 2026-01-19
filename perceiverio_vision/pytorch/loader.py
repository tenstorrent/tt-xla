# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PerceiverIO Vision model loader implementation for image classification
"""
import torch
from loguru import logger
from PIL import Image
from transformers import (
    AutoImageProcessor,
    PerceiverForImageClassificationConvProcessing,
    PerceiverForImageClassificationFourier,
    PerceiverForImageClassificationLearned,
)
from typing import Optional
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from ...base import ForgeModel
from ...tools.utils import print_compiled_model_results
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available PerceiverIO Vision model variants."""

    VISION_PERCEIVER_CONV = "deepmind/vision-perceiver-conv"
    VISION_PERCEIVER_LEARNED = "deepmind/vision-perceiver-learned"
    VISION_PERCEIVER_FOURIER = "deepmind/vision-perceiver-fourier"


class ModelLoader(ForgeModel):
    """PerceiverIO Vision model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.VISION_PERCEIVER_CONV: ModelConfig(
            pretrained_model_name="deepmind/vision-perceiver-conv",
        ),
        ModelVariant.VISION_PERCEIVER_LEARNED: ModelConfig(
            pretrained_model_name="deepmind/vision-perceiver-learned",
        ),
        ModelVariant.VISION_PERCEIVER_FOURIER: ModelConfig(
            pretrained_model_name="deepmind/vision-perceiver-fourier",
        ),
    }

    # Mapping of variants to their corresponding model classes
    _MODEL_CLASSES = {
        ModelVariant.VISION_PERCEIVER_CONV: PerceiverForImageClassificationConvProcessing,
        ModelVariant.VISION_PERCEIVER_LEARNED: PerceiverForImageClassificationLearned,
        ModelVariant.VISION_PERCEIVER_FOURIER: PerceiverForImageClassificationFourier,
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.VISION_PERCEIVER_CONV

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
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
            model="perceiverio_vision",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
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

        # Configuration parameters
        self.image_processor = None

    def load_model(self, dtype_override=None):
        """Load a PerceiverIO Vision model from HuggingFace."""

        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Get the appropriate model class for this variant
        model_class = self._MODEL_CLASSES[self._variant]

        # Load the model using the appropriate class
        model = model_class.from_pretrained(pretrained_model_name)
        model.eval()

        # Initialize image processor for this variant
        self.image_processor = AutoImageProcessor.from_pretrained(pretrained_model_name)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Generate sample inputs for PerceiverIO Vision models."""

        if self.image_processor is None:
            raise RuntimeError(
                "Model must be loaded first before loading inputs. Call load_model() first."
            )

        try:
            input_image = get_file(
                "http://images.cocodataset.org/val2017/000000039769.jpg"
            )
            image = Image.open(str(input_image))
            pixel_values = self.image_processor(
                images=image, return_tensors="pt"
            ).pixel_values
        except Exception as e:
            logger.warning(
                f"Failed to download the image file ({e}), replacing input with random tensor. "
                "Please check if the URL is up to date"
            )
            height = self.image_processor.to_dict()["size"]["height"]
            width = self.image_processor.to_dict()["size"]["width"]
            pixel_values = torch.rand(1, 3, height, width).to(torch.float32)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return pixel_values

    def print_cls_results(self, compiled_model_out):
        print_compiled_model_results(compiled_model_out)
