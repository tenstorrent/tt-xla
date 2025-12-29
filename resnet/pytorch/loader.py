# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ResNet model loader implementation
"""

from typing import Optional
from dataclasses import dataclass
from torchvision import models
import torch
import timm

from transformers import ResNetForImageClassification
from ...tools.utils import VisionPreprocessor, VisionPostprocessor


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
class ResNetConfig(ModelConfig):
    """Configuration specific to ResNet models"""

    source: ModelSource
    high_res_size: tuple = (
        None  # None means use default size, otherwise (width, height)
    )


class ModelVariant(StrEnum):
    """Available ResNet model variants."""

    # HuggingFace variants
    RESNET_50_HF = "resnet_50_hf"
    RESNET_50_HF_HIGH_RES = "resnet_50_hf_high_res"

    # TIMM variants
    RESNET_50_TIMM = "resnet50_timm"
    RESNET_50_TIMM_HIGH_RES = "resnet50_timm_high_res"

    # Torchvision variants
    RESNET_18 = "resnet18"
    RESNET_34 = "resnet34"
    RESNET_50 = "resnet50"
    RESNET_50_HIGH_RES = "resnet50_high_res"
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
        ModelVariant.RESNET_50_HF_HIGH_RES: ResNetConfig(
            pretrained_model_name="microsoft/resnet-50",
            source=ModelSource.HUGGING_FACE,
            high_res_size=(1280, 800),
        ),
        # TIMM variants
        ModelVariant.RESNET_50_TIMM: ResNetConfig(
            pretrained_model_name="resnet50",
            source=ModelSource.TIMM,
        ),
        ModelVariant.RESNET_50_TIMM_HIGH_RES: ResNetConfig(
            pretrained_model_name="resnet50",
            source=ModelSource.TIMM,
            high_res_size=(1280, 800),
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
        ModelVariant.RESNET_50_HIGH_RES: ResNetConfig(
            pretrained_model_name="resnet50",
            source=ModelSource.TORCHVISION,
            high_res_size=(1280, 800),
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

        if variant in [
            ModelVariant.RESNET_50_HF_HIGH_RES,
        ]:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="resnet",
            variant=variant,
            group=group,
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
                return name.replace("resnet", "ResNet") + "_Weights"

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
        """Load and return sample inputs (backward compatibility wrapper for input_preprocess).

        Args:
            dtype_override: Optional torch.dtype override.
            batch_size: Batch size (default: 1).
            image: Optional input image.

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        inputs = self.input_preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
        )
        return inputs if inputs.is_contiguous() else inputs.contiguous()

    def output_postprocess(
        self,
        output=None,
        co_out=None,
        framework_model=None,
        compiled_model=None,
        inputs=None,
        dtype_override=None,
    ):
        """Post-process model outputs.

        Args:
            output: Model output tensor (returns dict if provided).
            co_out: Compiled model outputs (legacy, prints results).
            framework_model: Original framework model (legacy).
            compiled_model: Compiled model (legacy).
            inputs: Input images (legacy).
            dtype_override: Optional dtype override (legacy).

        Returns:
            dict or None: Prediction dict if output provided, else None (prints results).
        """
        if self._postprocessor is None:
            model_name = self._variant_config.pretrained_model_name
            source = self._variant_config.source

            self._postprocessor = VisionPostprocessor(
                model_source=source,
                model_name=model_name,
                model_instance=self.model,
            )

        # New usage: return dict from output tensor
        if output is not None:
            return self._postprocessor.postprocess(output, top_k=1, return_dict=True)

        # Legacy usage: print results (backward compatibility)
        self._postprocessor.print_results(
            co_out=co_out,
            framework_model=framework_model,
            compiled_model=compiled_model,
            inputs=inputs,
            dtype_override=dtype_override,
        )
        return None
