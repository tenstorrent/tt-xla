# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ResNeXt model loader implementation
"""

import torch
from typing import Optional
from dataclasses import dataclass
from PIL import Image
from torchvision import transforms
from pytorchcv.model_provider import get_model as ptcv_get_model

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
from ...tools.utils import (
    VisionPreprocessor,
    VisionPostprocessor,
)


@dataclass
class ResNeXtConfig(ModelConfig):
    """Configuration specific to ResNeXt models"""

    source: ModelSource
    hub_source: Optional[str] = None  # Only used for torch_hub models


class ModelVariant(StrEnum):
    """Available ResNeXt model variants."""

    # Torch Hub variants
    RESNEXT50_32X4D = "resnext50_32x4d"
    RESNEXT101_32X8D = "resnext101_32x8d"
    RESNEXT101_64X4D = "resnext101_64x4d"
    RESNEXT101_32X8D_WSL = "resnext101_32x8d_wsl"

    # OSMR variants
    RESNEXT14_32X4D_OSMR = "resnext14_32x4d_osmr"
    RESNEXT26_32X4D_OSMR = "resnext26_32x4d_osmr"
    RESNEXT50_32X4D_OSMR = "resnext50_32x4d_osmr"
    RESNEXT101_64X4D_OSMR = "resnext101_64x4d_osmr"


class ModelLoader(ForgeModel):
    """ResNeXt model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # Torch Hub variants
        ModelVariant.RESNEXT50_32X4D: ResNeXtConfig(
            pretrained_model_name="resnext50_32x4d",
            source=ModelSource.TORCH_HUB,
            hub_source="pytorch/vision:v0.10.0",
        ),
        ModelVariant.RESNEXT101_32X8D: ResNeXtConfig(
            pretrained_model_name="resnext101_32x8d",
            source=ModelSource.TORCH_HUB,
            hub_source="pytorch/vision:v0.10.0",
        ),
        ModelVariant.RESNEXT101_64X4D: ResNeXtConfig(
            pretrained_model_name="resnext101_64x4d",
            source=ModelSource.TORCH_HUB,
            hub_source="pytorch/vision:v0.10.0",
        ),
        ModelVariant.RESNEXT101_32X8D_WSL: ResNeXtConfig(
            pretrained_model_name="resnext101_32x8d_wsl",
            source=ModelSource.TORCH_HUB,
            hub_source="facebookresearch/WSL-Images",
        ),
        # OSMR variants
        ModelVariant.RESNEXT14_32X4D_OSMR: ResNeXtConfig(
            pretrained_model_name="resnext14_32x4d",
            source=ModelSource.OSMR,
        ),
        ModelVariant.RESNEXT26_32X4D_OSMR: ResNeXtConfig(
            pretrained_model_name="resnext26_32x4d",
            source=ModelSource.OSMR,
        ),
        ModelVariant.RESNEXT50_32X4D_OSMR: ResNeXtConfig(
            pretrained_model_name="resnext50_32x4d",
            source=ModelSource.OSMR,
        ),
        ModelVariant.RESNEXT101_64X4D_OSMR: ResNeXtConfig(
            pretrained_model_name="resnext101_64x4d",
            source=ModelSource.OSMR,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.RESNEXT50_32X4D

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
            model="resnext",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the ResNeXt model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The ResNeXt model instance.
        """
        # Get the pretrained model name and source from the instance's variant config
        model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.TORCH_HUB:
            # Load model using torch.hub
            hub_source = self._variant_config.hub_source
            model = torch.hub.load(hub_source, model_name)
        elif source == ModelSource.OSMR:
            # Load model using pytorchcv
            model = ptcv_get_model(model_name, pretrained=True)

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

            # For TORCH_HUB and OSMR, use CUSTOM with standard ImageNet preprocessing
            if source == ModelSource.TORCH_HUB or source == ModelSource.OSMR:

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
                # For other sources (if any added in future)
                self._preprocessor = VisionPreprocessor(
                    model_source=source,
                    model_name=model_name,
                )

            if hasattr(self, "model") and self.model is not None:
                self._preprocessor.set_cached_model(self.model)

        return self._preprocessor.preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
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

            # For TORCH_HUB and OSMR, use TORCHVISION postprocessing (same ImageNet labels)
            postprocess_source = (
                ModelSource.TORCHVISION
                if source in (ModelSource.TORCH_HUB, ModelSource.OSMR)
                else source
            )

            self._postprocessor = VisionPostprocessor(
                model_source=postprocess_source,
                model_name=model_name,
                model_instance=self.model,
            )

        return self._postprocessor.postprocess(output, top_k=1, return_dict=True)
