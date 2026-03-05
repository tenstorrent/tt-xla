# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RegNet model loader implementation
"""

from typing import Optional
from dataclasses import dataclass
from torchvision import models
import torch

from transformers import RegNetForImageClassification
from ...tools.utils import VisionPreprocessor, VisionPostprocessor
from datasets import load_dataset

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


@dataclass
class RegNetConfig(ModelConfig):
    """Configuration specific to RegNet models"""

    source: ModelSource
    high_res_size: tuple = (
        None  # None means use default size, otherwise (width, height)
    )


class ModelVariant(StrEnum):
    """Available RegNet model variants."""

    # HuggingFace variants
    Y_040 = "Y_040"
    Y_064 = "Y_064"
    Y_080 = "Y_080"
    Y_120 = "Y_120"
    Y_160 = "Y_160"
    Y_320 = "Y_320"

    # Torchvision variants
    Y_400MF = "Y_400mf"
    Y_800MF = "Y_800mf"
    Y_1_6GF = "Y_1_6gf"
    Y_3_2GF = "Y_3_2gf"
    Y_8GF = "Y_8gf"
    Y_16GF = "Y_16gf"
    Y_32GF = "Y_32gf"
    Y_128GF = "Y_128gf"
    X_400MF = "X_400mf"
    X_800MF = "X_800mf"
    X_1_6GF = "X_1_6gf"
    X_3_2GF = "X_3_2gf"
    X_8GF = "X_8gf"
    X_16GF = "X_16gf"
    X_32GF = "X_32gf"


class ModelLoader(ForgeModel):
    """RegNet model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # HuggingFace variants
        ModelVariant.Y_040: RegNetConfig(
            pretrained_model_name="facebook/regnet-y-040",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.Y_064: RegNetConfig(
            pretrained_model_name="facebook/regnet-y-064",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.Y_080: RegNetConfig(
            pretrained_model_name="facebook/regnet-y-080",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.Y_120: RegNetConfig(
            pretrained_model_name="facebook/regnet-y-120",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.Y_160: RegNetConfig(
            pretrained_model_name="facebook/regnet-y-160",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.Y_320: RegNetConfig(
            pretrained_model_name="facebook/regnet-y-320",
            source=ModelSource.HUGGING_FACE,
        ),
        # Torchvision variants
        ModelVariant.Y_400MF: RegNetConfig(
            pretrained_model_name="regnet_y_400mf",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.Y_800MF: RegNetConfig(
            pretrained_model_name="regnet_y_800mf",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.Y_1_6GF: RegNetConfig(
            pretrained_model_name="regnet_y_1_6gf",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.Y_3_2GF: RegNetConfig(
            pretrained_model_name="regnet_y_3_2gf",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.Y_8GF: RegNetConfig(
            pretrained_model_name="regnet_y_8gf",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.Y_16GF: RegNetConfig(
            pretrained_model_name="regnet_y_16gf",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.Y_32GF: RegNetConfig(
            pretrained_model_name="regnet_y_32gf",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.Y_128GF: RegNetConfig(
            pretrained_model_name="regnet_y_128gf",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.X_400MF: RegNetConfig(
            pretrained_model_name="regnet_x_400mf",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.X_800MF: RegNetConfig(
            pretrained_model_name="regnet_x_800mf",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.X_1_6GF: RegNetConfig(
            pretrained_model_name="regnet_x_1_6gf",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.X_3_2GF: RegNetConfig(
            pretrained_model_name="regnet_x_3_2gf",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.X_8GF: RegNetConfig(
            pretrained_model_name="regnet_x_8gf",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.X_16GF: RegNetConfig(
            pretrained_model_name="regnet_x_16gf",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.X_32GF: RegNetConfig(
            pretrained_model_name="regnet_x_32gf",
            source=ModelSource.TORCHVISION,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.Y_040

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
            model="RegNet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the RegNet model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The RegNet model instance.
        """
        # Get the pretrained model name and source from the instance's variant config
        model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.HUGGING_FACE:
            # Load model from HuggingFace
            model = RegNetForImageClassification.from_pretrained(model_name, **kwargs)

        elif source == ModelSource.TORCHVISION:
            # Load model from torchvision
            # Get the weights class name (e.g., "regnet_y_400mf" -> "RegNet_Y_400MF_Weights")
            weight_class_name = model_name.upper() + "_Weights"
            weight_class_name = weight_class_name.replace("REGNET_", "RegNet_")

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
            high_res_size = self._variant_config.high_res_size

            def weight_class_name_fn(name: str) -> str:
                # Convert "regnet_y_400mf" -> "RegNet_Y_400MF_Weights"
                weight_class_name = name.upper() + "_Weights"
                return weight_class_name.replace("REGNET_", "RegNet_")

            self._preprocessor = VisionPreprocessor(
                model_source=source,
                model_name=model_name,
                high_res_size=high_res_size,
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
            image: Optional input image. If None, loads from HuggingFace datasets.

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        if image is None:
            dataset = load_dataset("huggingface/cats-image", split="test")
            image = dataset[0]["image"]
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
            dict: Prediction dictionary with top predictions.
        """
        if self._postprocessor is None:
            model_name = self._variant_config.pretrained_model_name
            source = self._variant_config.source

            self._postprocessor = VisionPostprocessor(
                model_source=source,
                model_name=model_name,
                model_instance=self.model,
            )

        return self._postprocessor.postprocess(output, top_k=1, return_dict=True)
