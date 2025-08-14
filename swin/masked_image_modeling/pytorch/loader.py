# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Swin model loader implementation for masked image modeling.
"""

from transformers import Swinv2ForMaskedImageModeling, ViTImageProcessor
from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel
from PIL import Image
from third_party.tt_forge_models.tools.utils import get_file
from typing import Optional


class ModelVariant(StrEnum):
    """Available Swin model variants for masked image modeling."""

    SWINV2_TINY = "microsoft/swinv2-tiny-patch4-window8-256"


class ModelLoader(ForgeModel):
    """Swin model loader implementation for masked image modeling."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.SWINV2_TINY: ModelConfig(
            pretrained_model_name="microsoft/swinv2-tiny-patch4-window8-256",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.SWINV2_TINY

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Get the pretrained model name from the instance's variant config
        model_name = self._variant_config.pretrained_model_name
        self.model_name = model_name
        self.feature_extractor = None
        self.model = None

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
            model="swin-masked-lm",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load Swin model for masked image modeling from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Swin model instance.
        """
        # Initialize feature extractor
        self.feature_extractor = ViTImageProcessor.from_pretrained(self.model_name)

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = Swinv2ForMaskedImageModeling.from_pretrained(
            self.model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Swin masked image modeling model.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor : Input tensors that can be fed to the model.
        """
        if self.feature_extractor is None:
            # Ensure feature extractor is initialized
            self.load_model(dtype_override=dtype_override)

        # Load image
        image_path = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_path)

        # Preprocess image
        inputs = self.feature_extractor(images=image, return_tensors="pt").pixel_values

        # Replicate tensors for batch size
        inputs = inputs.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
