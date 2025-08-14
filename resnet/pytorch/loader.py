# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ResNet model loader implementation
"""

from typing import Optional
from dataclasses import dataclass
from PIL import Image
from torchvision import models
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from transformers import ResNetForImageClassification, AutoImageProcessor
from ...tools.utils import get_file, print_compiled_model_results


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
from .src.utils import run_and_print_results


@dataclass
class ResNetConfig(ModelConfig):
    """Configuration specific to ResNet models"""

    source: ModelSource


class ModelVariant(StrEnum):
    """Available ResNet model variants."""

    # HuggingFace variants
    RESNET_50_HF = "resnet_50_hf"

    # TIMM variants
    RESNET_50_TIMM = "resnet50"

    # Torchvision variants
    RESNET_18 = "resnet18"
    RESNET_34 = "resnet34"
    RESNET_50 = "resnet50"
    RESNET_101 = "resnet101"
    RESNET_152 = "resnet152"


class ModelLoader(ForgeModel):
    """ResNet model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # HuggingFace variants
        ModelVariant.RESNET_50_HF: ResNetConfig(
            pretrained_model_name="microsoft/resnet-50",
            source=ModelSource.HUGGING_FACE,
        ),
        # TIMM variants
        ModelVariant.RESNET_50_TIMM: ResNetConfig(
            pretrained_model_name="resnet50",
            source=ModelSource.TIMM,
        ),
        # Torchvision variants
        ModelVariant.RESNET_18: ResNetConfig(
            pretrained_model_name="resnet18",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.RESNET_34: ResNetConfig(
            pretrained_model_name="resnet34",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.RESNET_50: ResNetConfig(
            pretrained_model_name="resnet50",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.RESNET_101: ResNetConfig(
            pretrained_model_name="resnet101",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.RESNET_152: ResNetConfig(
            pretrained_model_name="resnet152",
            source=ModelSource.TORCHVISION,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.RESNET_50_HF

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.image_processor = None
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
            model="resnet",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the ResNet model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The ResNet model instance.
        """
        # Get the pretrained model name and source from the instance's variant config
        model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.HUGGING_FACE:
            # Load model from HuggingFace
            model = ResNetForImageClassification.from_pretrained(model_name)

        elif source == ModelSource.TIMM:
            # Load model using timm
            model = timm.create_model(model_name, pretrained=True)

        elif source == ModelSource.TORCHVISION:
            # Load model from torchvision
            # Get the weights class name (e.g., "resnet50" -> "ResNet50_Weights")
            weight_class_name = model_name.replace("resnet", "ResNet") + "_Weights"

            # Get the weights class and model function
            weights = getattr(models, weight_class_name).DEFAULT
            model_func = getattr(models, model_name)
            model = model_func(weights=weights)

        model.eval()

        # Store model for potential use in load_inputs and post_processing
        self.model = model

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the ResNet model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for ResNet.
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
                self.image_processor = AutoImageProcessor.from_pretrained(model_name)

            # Preprocess image using HuggingFace image processor
            inputs = self.image_processor(
                images=image, return_tensors="pt"
            ).pixel_values

        elif source == ModelSource.TIMM:

            # Use cached model if available, otherwise load it
            if hasattr(self, "model") and self.model is not None:
                model_for_config = self.model
            else:
                model_for_config = self.load_model(dtype_override)

            # Preprocess image using model's data config
            data_config = resolve_data_config({}, model=model_for_config)
            timm_transforms = create_transform(**data_config)
            inputs = timm_transforms(image).unsqueeze(0)

        elif source == ModelSource.TORCHVISION:

            # Get the weights class name for torchvision preprocessing
            weight_class_name = model_name.replace("resnet", "ResNet") + "_Weights"

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

    def post_process(
        self, co_out=None, framework_model=None, compiled_model=None, inputs=None
    ):

        """
        Post-processes model outputs based on the model source.

        Args:
            co_out : Outputs from the compiled model
            framework_model: The original framework-based model.
            compiled_model: The compiled version of the model.
            inputs: A list of images to process and classify.

        Returns:
            None: Prints predicted results.
        """

        source = self._variant_config.source

        if source == ModelSource.HUGGING_FACE:
            run_and_print_results(framework_model, compiled_model, inputs)
        else:
            print_compiled_model_results(co_out)
