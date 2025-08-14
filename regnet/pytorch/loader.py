# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RegNet model loader implementation
"""

from typing import Optional
from dataclasses import dataclass
from PIL import Image
from torchvision import models, transforms
import torch
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
from transformers import AutoFeatureExtractor, RegNetForImageClassification
from ...tools.utils import get_file, print_compiled_model_results


@dataclass
class RegNetConfig(ModelConfig):
    """Configuration specific to RegNet models"""

    source: ModelSource


class ModelVariant(StrEnum):
    """Available RegNet model variants."""

    # HuggingFace variants
    Y_040 = "regnet_y_040"
    Y_064 = "regnet_y_064"
    Y_080 = "regnet_y_080"
    Y_120 = "regnet_y_120"
    Y_160 = "regnet_y_160"
    Y_320 = "regnet_y_320"

    # Torchvision variants
    Y_400MF = "regnet_y_400mf"
    Y_800MF = "regnet_y_800mf"
    Y_1_6GF = "regnet_y_1_6gf"
    Y_3_2GF = "regnet_y_3_2gf"
    Y_8GF = "regnet_y_8gf"
    Y_16GF = "regnet_y_16gf"
    Y_32GF = "regnet_y_32gf"
    Y_128GF = "regnet_y_128gf"
    X_400MF = "regnet_x_400mf"
    X_800MF = "regnet_x_800mf"
    X_1_6GF = "regnet_x_1_6gf"
    X_3_2GF = "regnet_x_3_2gf"
    X_8GF = "regnet_x_8gf"
    X_16GF = "regnet_x_16gf"
    X_32GF = "regnet_x_32gf"


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

        # Get source from variant config
        source = cls._VARIANTS[variant].source

        return ModelInfo(
            model="regnet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
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
            model = RegNetForImageClassification.from_pretrained(model_name)

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

        # Store model for potential use in post_processing
        self.model = model

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the RegNet model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for RegNet.
        """
        # Get the pretrained model name and source from the instance's variant config
        model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        # Get the Image
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file).convert("RGB")

        if source == ModelSource.HUGGING_FACE:
            # Initialize feature extractor if not already done
            if self.feature_extractor is None:
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                    model_name
                )

            # Preprocess image using HuggingFace feature extractor
            inputs = self.feature_extractor(
                images=image, return_tensors="pt"
            ).pixel_values

        elif source == ModelSource.TORCHVISION:
            # Get the weights class name for torchvision preprocessing
            weight_class_name = model_name.upper() + "_Weights"
            weight_class_name = weight_class_name.replace("REGNET_", "RegNet_")

            # Get the weights and use their transforms
            weights = getattr(models, weight_class_name).DEFAULT
            preprocess = weights.transforms()
            inputs = preprocess(image).unsqueeze(0)

        # Replicate tensors for batch size
        inputs = inputs.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs

    def post_processing(self, co_out):
        """Post-process the model outputs based on source.

        Args:
            co_out: Compiled model outputs

        Returns:
            None: Prints the predicted class
        """
        source = self._variant_config.source

        if source == ModelSource.HUGGING_FACE:
            logits = co_out[0]
            predicted_label = logits.argmax(-1).item()
            print("Predicted class:", self.model.config.id2label[predicted_label])

        elif source == ModelSource.TORCHVISION:
            print_compiled_model_results(co_out)
