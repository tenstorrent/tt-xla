# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VovNet model loader implementation
"""
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms

from pytorchcv.model_provider import get_model as ptcv_get_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from .src.utils import download_model, preprocess_steps, preprocess_timm_model
from .src.src_vovnet_stigma import (
    vovnet39 as stigma_vovnet39,
    vovnet57 as stigma_vovnet57,
)
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
    print_compiled_model_results,
)
from dataclasses import dataclass
from loguru import logger


@dataclass
class VovNetConfig(ModelConfig):
    source: ModelSource
    high_res_size: tuple = (
        None  # None means use default size, otherwise (width, height)
    )


class ModelVariant(StrEnum):
    """Available VovNet model variants."""

    # OSMR (pytorchcv) image classification variants
    VOVNET27S = "vovnet27s"
    VOVNET39 = "vovnet39"
    VOVNET57 = "vovnet57"

    # TorchHub variant
    VOVNET39_TORCHHUB = "vovnet39_th"
    VOVNET57_TORCHHUB = "vovnet57_th"

    # TIMM image classification variants (subset)
    TIMM_VOVNET19B_DW = "ese_vovnet19b_dw"
    TIMM_VOVNET39B = "ese_vovnet39b"
    TIMM_VOVNET99B = "ese_vovnet99b"
    TIMM_VOVNET19B_DW_RAIN1K = "ese_vovnet19b_dw.ra_in1k"


class ModelLoader(ForgeModel):
    """VovNet model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # OSMR variants
        ModelVariant.VOVNET27S: VovNetConfig(
            pretrained_model_name="vovnet27s", source=ModelSource.OSMR
        ),
        ModelVariant.VOVNET39: VovNetConfig(
            pretrained_model_name="vovnet39", source=ModelSource.OSMR
        ),
        ModelVariant.VOVNET57: VovNetConfig(
            pretrained_model_name="vovnet57", source=ModelSource.OSMR
        ),
        # TorchHub
        ModelVariant.VOVNET39_TORCHHUB: VovNetConfig(
            pretrained_model_name="vovnet39", source=ModelSource.TORCH_HUB
        ),
        ModelVariant.VOVNET57_TORCHHUB: VovNetConfig(
            pretrained_model_name="vovnet57", source=ModelSource.TORCH_HUB
        ),
        # TIMM variants
        ModelVariant.TIMM_VOVNET19B_DW: VovNetConfig(
            pretrained_model_name="ese_vovnet19b_dw", source=ModelSource.TIMM
        ),
        ModelVariant.TIMM_VOVNET39B: VovNetConfig(
            pretrained_model_name="ese_vovnet39b", source=ModelSource.TIMM
        ),
        ModelVariant.TIMM_VOVNET99B: VovNetConfig(
            pretrained_model_name="ese_vovnet99b", source=ModelSource.TIMM
        ),
        ModelVariant.TIMM_VOVNET19B_DW_RAIN1K: VovNetConfig(
            pretrained_model_name="ese_vovnet19b_dw.ra_in1k", source=ModelSource.TIMM
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.VOVNET27S

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.input_shape = (3, 224, 224)
        self.model = None
        self._preprocessor = None
        self._postprocessor = None

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

        if variant in [ModelVariant.TIMM_VOVNET19B_DW_RAIN1K]:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="vovnet",
            variant=variant,
            group=group,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load a VovNet model based on the configured source for this variant."""
        cfg = self._variant_config
        model_name = cfg.pretrained_model_name
        source = cfg.source

        if source == ModelSource.OSMR:
            model = ptcv_get_model(model_name, pretrained=True)
        elif source == ModelSource.TIMM:
            model, _ = download_model(preprocess_timm_model, model_name)
        elif source == ModelSource.TORCH_HUB:
            model_fn = stigma_vovnet39 if "39" in model_name else stigma_vovnet57
            model, _ = download_model(preprocess_steps, model_fn)
        else:
            raise ValueError(f"Unsupported source: {source}")

        model.eval()

        # Store model for potential use in input preprocessing and postprocessing
        self.model = model

        # Update preprocessor with cached model (for TIMM models)
        if self._preprocessor is not None:
            self._preprocessor.set_cached_model(model)

        # Update postprocessor with model instance (for HuggingFace models)
        if self._postprocessor is not None:
            self._postprocessor.set_model_instance(model)

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

            # For OSMR and TORCH_HUB, use CUSTOM with standard ImageNet preprocessing
            if source == ModelSource.OSMR or source == ModelSource.TORCH_HUB:

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
                # TIMM source
                self._preprocessor = VisionPreprocessor(
                    model_source=source,
                    model_name=model_name,
                    high_res_size=high_res_size,
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
        return self.input_preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
        )

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

            # For OSMR and TORCH_HUB, use TORCHVISION postprocessing (same ImageNet labels)
            if source == ModelSource.OSMR or source == ModelSource.TORCH_HUB:
                postprocess_source = ModelSource.TORCHVISION
            else:
                postprocess_source = source

            self._postprocessor = VisionPostprocessor(
                model_source=postprocess_source,
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

    def print_cls_results(self, compiled_model_out):
        """Legacy method for backward compatibility."""
        print_compiled_model_results(compiled_model_out)
