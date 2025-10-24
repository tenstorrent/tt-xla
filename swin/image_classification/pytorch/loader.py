# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Swin model loader implementation
"""

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from torchvision import models
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, ViTImageProcessor
from ....tools.utils import get_file, print_compiled_model_results
from typing import Optional
from dataclasses import dataclass


@dataclass
class SwinConfig(ModelConfig):
    """Configuration specific to Swin models"""

    source: ModelSource


class ModelVariant(StrEnum):
    """Available Swin model variants."""

    # HuggingFace variants
    SWIN_TINY_HF = "microsoft/swin-tiny-patch4-window7-224"
    SWINV2_TINY_HF = "microsoft/swinv2-tiny-patch4-window8-256"

    # Torchvision variants
    SWIN_T = "swin_t"
    SWIN_S = "swin_s"
    SWIN_B = "swin_b"
    SWIN_V2_T = "swin_v2_t"
    SWIN_V2_S = "swin_v2_s"
    SWIN_V2_B = "swin_v2_b"


class ModelLoader(ForgeModel):
    """Swin model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # HuggingFace variants
        ModelVariant.SWIN_TINY_HF: SwinConfig(
            pretrained_model_name="microsoft/swin-tiny-patch4-window7-224",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.SWINV2_TINY_HF: SwinConfig(
            pretrained_model_name="microsoft/swinv2-tiny-patch4-window8-256",
            source=ModelSource.HUGGING_FACE,
        ),
        # Torchvision variants
        ModelVariant.SWIN_T: SwinConfig(
            pretrained_model_name="swin_t",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.SWIN_S: SwinConfig(
            pretrained_model_name="swin_s",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.SWIN_B: SwinConfig(
            pretrained_model_name="swin_b",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.SWIN_V2_T: SwinConfig(
            pretrained_model_name="swin_v2_t",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.SWIN_V2_S: SwinConfig(
            pretrained_model_name="swin_v2_s",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.SWIN_V2_B: SwinConfig(
            pretrained_model_name="swin_v2_b",
            source=ModelSource.TORCHVISION,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.SWIN_S

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

        # Get source from variant config
        source = cls._VARIANTS[variant].source

        return ModelInfo(
            model="swin",
            variant=variant,
            group=ModelGroup.RED
            if variant == ModelVariant.SWIN_S
            else ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.image_processor = None
        self._weights = None  # For torchvision models

    def load_model(self, dtype_override=None):
        """Load and return the Swin model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Swin model instance.
        """
        # Get the pretrained model name and source from the instance's variant config
        model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.HUGGING_FACE:
            # Load model from HuggingFace
            model = AutoModelForImageClassification.from_pretrained(model_name)

        elif source == ModelSource.TORCHVISION:
            # Load model from torchvision
            # Get the weights class name (e.g., "swin_t" -> "Swin_T_Weights")
            weight_class_name = model_name.upper() + "_Weights"
            weight_class_name = weight_class_name.replace("SWIN_", "Swin_")

            # Get the weights class and model function
            weights = getattr(models, weight_class_name).DEFAULT
            model_func = getattr(models, model_name)
            model = model_func(weights=weights)
            self._weights = weights

        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Swin model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for Swin.
        """
        # Get the pretrained model name and source from the instance's variant config
        model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        # Get the Image
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file).convert("RGB")

        if source == ModelSource.HUGGING_FACE:
            # Initialize image processor if not already done
            if self.image_processor is None:
                self.image_processor = ViTImageProcessor.from_pretrained(model_name)

            # Preprocess image using HuggingFace image processor
            inputs = self.image_processor(
                images=image, return_tensors="pt"
            ).pixel_values
        elif source == ModelSource.TORCHVISION:
            # Use torchvision preprocessing
            if self._weights is None:
                # Need to load weights if not already loaded
                # Get the weights class name (e.g., "swin_t" -> "Swin_T_Weights")
                weight_class_name = model_name.upper() + "_Weights"
                weight_class_name = weight_class_name.replace("SWIN_", "Swin_")
                self._weights = getattr(models, weight_class_name).DEFAULT

            preprocess = self._weights.transforms()
            img_t = preprocess(image)
            inputs = torch.unsqueeze(img_t, 0).contiguous()

        # Replicate tensors for batch size
        inputs = inputs.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs

    def print_cls_results(self, compiled_model_out):
        print_compiled_model_results(compiled_model_out)
