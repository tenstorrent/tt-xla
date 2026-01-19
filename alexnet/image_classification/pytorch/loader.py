# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AlexNet model loader implementation
"""

from typing import Optional
from dataclasses import dataclass
from torchvision import models
import torch
from pytorchcv.model_provider import get_model as ptcv_get_model

from ....tools.utils import VisionPreprocessor, VisionPostprocessor

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


@dataclass
class AlexNetConfig(ModelConfig):
    """Configuration specific to AlexNet models"""

    source: ModelSource
    high_res_size: tuple = (
        None  # None means use default size, otherwise (width, height)
    )


class ModelVariant(StrEnum):
    """Available AlexNet model variants."""

    # Torchvision variants
    ALEXNET = "alexnet"
    ALEXNET_HIGH_RES = "alexnet_high_res"

    # OSMR variant
    ALEXNET_OSMR_B = "alexnetb"


class ModelLoader(ForgeModel):
    """AlexNet model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # Torchvision variants
        ModelVariant.ALEXNET: AlexNetConfig(
            pretrained_model_name="alexnet",
            source=ModelSource.TORCHVISION,
        ),
        ModelVariant.ALEXNET_HIGH_RES: AlexNetConfig(
            pretrained_model_name="alexnet",
            source=ModelSource.TORCHVISION,
            high_res_size=(1280, 800),
        ),
        # OSMR variant
        ModelVariant.ALEXNET_OSMR_B: AlexNetConfig(
            pretrained_model_name="alexnetb",
            source=ModelSource.OSMR,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.ALEXNET

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
            model="alexnet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the AlexNet model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The AlexNet model instance.
        """
        # Get the pretrained model name and source from the instance's variant config
        model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.OSMR:
            # Load model from OSMR
            model = ptcv_get_model(model_name, pretrained=True)
        elif source == ModelSource.TORCHVISION:
            # Load model from torchvision
            # Get the weights class name (e.g., "alexnet" -> "AlexNet_Weights")
            weight_class_name = model_name.replace("alexnet", "AlexNet") + "_Weights"

            # Get the weights class and model function
            weights = getattr(models, weight_class_name).DEFAULT
            model_func = getattr(models, model_name)
            model = model_func(weights=weights)
        else:
            raise ValueError(f"Unsupported model source: {source}")

        model.eval()

        # Store model for potential use in input preprocessing and postprocessing
        self.model = model

        # Update preprocessor with cached model (for TIMM models - not used for AlexNet)
        if self._preprocessor is not None:
            self._preprocessor.set_cached_model(model)

        # Update postprocessor with model instance (for HuggingFace models - not used for AlexNet)
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
                return name.replace("alexnet", "AlexNet") + "_Weights"

            # For OSMR source, use custom preprocessing
            if source == ModelSource.OSMR:
                from PIL import Image
                from torchvision import transforms

                def custom_preprocess_fn(img: Image.Image) -> torch.Tensor:
                    preprocess = transforms.Compose(
                        [
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                            ),
                        ]
                    )
                    return preprocess(img)

                self._preprocessor = VisionPreprocessor(
                    model_source=ModelSource.CUSTOM,
                    model_name=model_name,
                    high_res_size=high_res_size,
                    custom_preprocess_fn=custom_preprocess_fn,
                )
            else:
                self._preprocessor = VisionPreprocessor(
                    model_source=source,
                    model_name=model_name,
                    high_res_size=high_res_size,
                    weight_class_name_fn=(
                        weight_class_name_fn
                        if source == ModelSource.TORCHVISION
                        else None
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
            dict: Prediction dictionary with top predictions.
        """
        if self._postprocessor is None:
            model_name = self._variant_config.pretrained_model_name
            source = self._variant_config.source

            # For OSMR, use TORCHVISION postprocessing (same ImageNet labels)
            postprocess_source = (
                ModelSource.TORCHVISION if source == ModelSource.OSMR else source
            )

            self._postprocessor = VisionPostprocessor(
                model_source=postprocess_source,
                model_name=model_name,
                model_instance=self.model,
            )

        return self._postprocessor.postprocess(output, top_k=1, return_dict=True)
