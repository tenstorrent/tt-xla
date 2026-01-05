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
from typing import Optional
from dataclasses import dataclass
from transformers import AutoModelForImageClassification

from ....tools.utils import VisionPreprocessor, VisionPostprocessor


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
        self.model = None
        self._preprocessor = None
        self._postprocessor = None

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

        model.eval()

        # Store model for potential use in input preprocessing and postprocessing
        self.model = model

        # Update preprocessor with cached model (for TIMM models)
        if self._preprocessor is not None:
            self._preprocessor.set_cached_model(model)

        # Update postprocessor with model instance (for HuggingFace models)
        if self._postprocessor is not None:
            self._postprocessor.set_model_instance(model)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def input_preprocess(self, dtype_override=None, batch_size=1, image=None):
        """Preprocess input image(s) and return model-ready input tensor.

        Args:
            dtype_override: Optional torch.dtype override (default: float32).
            batch_size: Batch size (ignored if image is a list).
            image: PIL Image, URL string, tensor, list of images/URLs, or None (uses default COCO image).

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        if self._preprocessor is None:
            model_name = self._variant_config.pretrained_model_name
            source = self._variant_config.source

            def weight_class_name_fn(name: str) -> str:
                return name.upper().replace("SWIN_", "Swin_") + "_Weights"

            self._preprocessor = VisionPreprocessor(
                model_source=source,
                model_name=model_name,
                weight_class_name_fn=(
                    weight_class_name_fn if source == ModelSource.TORCHVISION else None
                ),
            )

            if hasattr(self, "model") and self.model is not None:
                self._preprocessor.set_cached_model(self.model)

        model_for_config = None
        if self._variant_config.source == ModelSource.TIMM:
            if hasattr(self, "model") and self.model is not None:
                model_for_config = self.model

        return self._preprocessor.preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
            model_for_config=model_for_config,
        )

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        """Load and return sample inputs for the model.

        Args:
            dtype_override: Optional torch.dtype override.
            batch_size: Batch size (default: 1).
            image: Optional input image.

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        return self.input_preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
        )

    def output_postprocess(self, output):
        """Post-process model outputs.

        Args:
            output: Model output tensor.

        Returns:
            dict: Prediction dict with top predictions.
        """
        if self._postprocessor is None:
            model_name = self._variant_config.pretrained_model_name
            source = self._variant_config.source

            def weight_class_name_fn(name: str) -> str:
                return name.upper().replace("SWIN_", "Swin_") + "_Weights"

            self._postprocessor = VisionPostprocessor(
                model_source=source,
                model_name=model_name,
                model_instance=self.model,
                weight_class_name_fn=(
                    weight_class_name_fn if source == ModelSource.TORCHVISION else None
                ),
            )

        return self._postprocessor.postprocess(output, top_k=1, return_dict=True)
