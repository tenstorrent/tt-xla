# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Deit model loader implementation
"""

from typing import Optional
from PIL import Image
from transformers import AutoFeatureExtractor, ViTForImageClassification

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
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available DeiT model variants."""

    BASE = "base"
    BASE_DISTILLED = "base_distilled"
    SMALL = "small"
    TINY = "tiny"


class ModelLoader(ForgeModel):
    """Deit model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="facebook/deit-base-patch16-224",
        ),
        ModelVariant.BASE_DISTILLED: ModelConfig(
            pretrained_model_name="facebook/deit-base-distilled-patch16-224",
        ),
        ModelVariant.SMALL: ModelConfig(
            pretrained_model_name="facebook/deit-small-patch16-224",
        ),
        ModelVariant.TINY: ModelConfig(
            pretrained_model_name="facebook/deit-tiny-patch16-224",
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
        self.feature_extractor = None

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
            model="deit",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the Deit model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Deit model instance.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Initialize feature extractor first
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            pretrained_model_name
        )

        # Load pre-trained model from HuggingFace
        model = ViTForImageClassification.from_pretrained(pretrained_model_name)

        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Deit model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Preprocessed input tensors suitable for ViTForImageClassification.
        """

        # Ensure feature extractor is initialized
        if self.feature_extractor is None:
            self.load_model(
                dtype_override=dtype_override
            )  # This will initialize the feature extractor

        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(str(image_file))
        inputs = self.feature_extractor(images=image, return_tensors="pt")

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            for key in inputs:
                inputs[key] = inputs[key].to(dtype_override)

        return inputs

    def post_processing(self, co_out, model):
        """Post-process the model outputs.

        Args:
            co_out: Compiled model outputs
            model: The model instance for accessing configuration

        Returns:
            None: Prints the predicted class
        """
        logits = co_out[0]
        predicted_class_idx = logits.argmax(-1).item()
        print("Predicted class:", model.config.id2label[predicted_class_idx])
