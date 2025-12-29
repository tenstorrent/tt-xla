# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
WideResnet model loader implementation
"""
import torch
from PIL import Image
from typing import Optional
from torchvision import transforms
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
from ...tools.utils import (
    VisionPreprocessor,
    VisionPostprocessor,
)
from dataclasses import dataclass
import timm


@dataclass
class WideResnetConfig(ModelConfig):
    source: ModelSource


class ModelVariant(StrEnum):
    """Available WideResnet model variants."""

    # Torch Hub/Torchvision variants
    WIDE_RESNET50_2 = "wide_resnet50_2"
    WIDE_RESNET101_2 = "wide_resnet101_2"

    # TIMM variants
    TIMM_WIDE_RESNET50_2 = "wide_resnet50_2.timm"
    TIMM_WIDE_RESNET101_2 = "wide_resnet101_2.timm"


class ModelLoader(ForgeModel):
    """WideResnet model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # Torch Hub variants
        ModelVariant.WIDE_RESNET50_2: WideResnetConfig(
            pretrained_model_name="wide_resnet50_2",
            source=ModelSource.TORCH_HUB,
        ),
        ModelVariant.WIDE_RESNET101_2: WideResnetConfig(
            pretrained_model_name="wide_resnet101_2",
            source=ModelSource.TORCH_HUB,
        ),
        # TIMM variants
        ModelVariant.TIMM_WIDE_RESNET50_2: WideResnetConfig(
            pretrained_model_name="wide_resnet50_2",
            source=ModelSource.TIMM,
        ),
        ModelVariant.TIMM_WIDE_RESNET101_2: WideResnetConfig(
            pretrained_model_name="wide_resnet101_2",
            source=ModelSource.TIMM,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.WIDE_RESNET50_2

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
        source = cls._VARIANTS[variant].source
        return ModelInfo(
            model="wide_resnet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.model = None
        self._preprocessor = None
        self._postprocessor = None

    def load_model(self, dtype_override=None):
        """Load a WideResnet model from Torch Hub or TIMM depending on variant source."""

        # Get the pretrained model name and source from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.TIMM:
            model = timm.create_model(pretrained_model_name, pretrained=True)
        else:
            model = torch.hub.load(
                "pytorch/vision:v0.10.0", pretrained_model_name, pretrained=True
            )
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

            # For TORCH_HUB, use CUSTOM with standard ImageNet preprocessing
            if source == ModelSource.TORCH_HUB:

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
                    custom_preprocess_fn=custom_preprocess_fn,
                )
            else:
                # TIMM source
                self._preprocessor = VisionPreprocessor(
                    model_source=source,
                    model_name=model_name,
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
            dict: Prediction dict with top-k results.
        """
        if self._postprocessor is None:
            model_name = self._variant_config.pretrained_model_name
            source = self._variant_config.source

            # For TORCH_HUB, use TORCHVISION postprocessing (same ImageNet labels)
            postprocess_source = (
                ModelSource.TORCHVISION if source == ModelSource.TORCH_HUB else source
            )

            self._postprocessor = VisionPostprocessor(
                model_source=postprocess_source,
                model_name=model_name,
                model_instance=self.model,
            )

        return self._postprocessor.postprocess(output, top_k=1, return_dict=True)
