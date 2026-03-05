# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HRNet model loader implementation
"""

import timm
from typing import Optional
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
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
from datasets import load_dataset
from ...tools.utils import print_compiled_model_results
from dataclasses import dataclass


@dataclass
class HRNetConfig(ModelConfig):
    """Configuration specific to HRNet models"""

    source: ModelSource = ModelSource.TIMM


class ModelVariant(StrEnum):
    """Available HRNet model variants."""

    # TIMM variants
    HRNET_W18_SMALL = "W18_Small"
    HRNET_W18_SMALL_V2 = "W18_Small_v2"
    HRNET_W18 = "W18"
    HRNET_W30 = "W30"
    HRNET_W32 = "W32"
    HRNET_W40 = "W40"
    HRNET_W44 = "W44"
    HRNET_W48 = "W48"
    HRNET_W64 = "W64"
    HRNET_W18_MS_AUG_IN1K = "W18.ms_Aug_In1k"

    # OSMR (pytorchcv) variants
    HRNET_W18_SMALL_V1_OSMR = "W18_Small_v1_Osmr"
    HRNET_W18_SMALL_V2_OSMR = "W18_Small_v2_Osmr"
    HRNETV2_W18_OSMR = "v2_W18_Osmr"
    HRNETV2_W30_OSMR = "v2_W30_Osmr"
    HRNETV2_W32_OSMR = "v2_W32_Osmr"
    HRNETV2_W40_OSMR = "v2_W40_Osmr"
    HRNETV2_W44_OSMR = "v2_W44_Osmr"
    HRNETV2_W48_OSMR = "v2_W48_Osmr"
    HRNETV2_W64_OSMR = "v2_W64_Osmr"


class ModelLoader(ForgeModel):
    """HRNet model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # TIMM variants
        ModelVariant.HRNET_W18_SMALL: HRNetConfig(
            pretrained_model_name="hrnet_w18_small",
            source=ModelSource.TIMM,
        ),
        ModelVariant.HRNET_W18_SMALL_V2: HRNetConfig(
            pretrained_model_name="hrnet_w18_small_v2",
            source=ModelSource.TIMM,
        ),
        ModelVariant.HRNET_W18: HRNetConfig(
            pretrained_model_name="hrnet_w18",
            source=ModelSource.TIMM,
        ),
        ModelVariant.HRNET_W30: HRNetConfig(
            pretrained_model_name="hrnet_w30",
            source=ModelSource.TIMM,
        ),
        ModelVariant.HRNET_W32: HRNetConfig(
            pretrained_model_name="hrnet_w32",
            source=ModelSource.TIMM,
        ),
        ModelVariant.HRNET_W40: HRNetConfig(
            pretrained_model_name="hrnet_w40",
            source=ModelSource.TIMM,
        ),
        ModelVariant.HRNET_W44: HRNetConfig(
            pretrained_model_name="hrnet_w44",
            source=ModelSource.TIMM,
        ),
        ModelVariant.HRNET_W48: HRNetConfig(
            pretrained_model_name="hrnet_w48",
            source=ModelSource.TIMM,
        ),
        ModelVariant.HRNET_W64: HRNetConfig(
            pretrained_model_name="hrnet_w64",
            source=ModelSource.TIMM,
        ),
        ModelVariant.HRNET_W18_MS_AUG_IN1K: HRNetConfig(
            pretrained_model_name="hrnet_w18.ms_aug_in1k",
            source=ModelSource.TIMM,
        ),
        # OSMR variants
        ModelVariant.HRNET_W18_SMALL_V1_OSMR: HRNetConfig(
            pretrained_model_name="hrnet_w18_small_v1",
            source=ModelSource.OSMR,
        ),
        ModelVariant.HRNET_W18_SMALL_V2_OSMR: HRNetConfig(
            pretrained_model_name="hrnet_w18_small_v2",
            source=ModelSource.OSMR,
        ),
        ModelVariant.HRNETV2_W18_OSMR: HRNetConfig(
            pretrained_model_name="hrnetv2_w18",
            source=ModelSource.OSMR,
        ),
        ModelVariant.HRNETV2_W30_OSMR: HRNetConfig(
            pretrained_model_name="hrnetv2_w30",
            source=ModelSource.OSMR,
        ),
        ModelVariant.HRNETV2_W32_OSMR: HRNetConfig(
            pretrained_model_name="hrnetv2_w32",
            source=ModelSource.OSMR,
        ),
        ModelVariant.HRNETV2_W40_OSMR: HRNetConfig(
            pretrained_model_name="hrnetv2_w40",
            source=ModelSource.OSMR,
        ),
        ModelVariant.HRNETV2_W44_OSMR: HRNetConfig(
            pretrained_model_name="hrnetv2_w44",
            source=ModelSource.OSMR,
        ),
        ModelVariant.HRNETV2_W48_OSMR: HRNetConfig(
            pretrained_model_name="hrnetv2_w48",
            source=ModelSource.OSMR,
        ),
        ModelVariant.HRNETV2_W64_OSMR: HRNetConfig(
            pretrained_model_name="hrnetv2_w64",
            source=ModelSource.OSMR,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.HRNET_W18_SMALL

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

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
        source = cls._VARIANTS[variant].source
        return ModelInfo(
            model="HRNet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_KEYPOINT_DET,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load pretrained HRNet model for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The HRNet model instance.
        """
        # Get the pretrained model name from the instance's variant config
        model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        # Load model using appropriate source
        if source == ModelSource.TIMM:
            model = timm.create_model(model_name, pretrained=True)
        else:
            model = ptcv_get_model(model_name, pretrained=True)

        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        # Store model for use in load_inputs (to avoid reloading)
        self._cached_model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample input for HRNet model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for HRNet.
        """
        source = self._variant_config.source

        # Load image from HuggingFace dataset
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"].convert("RGB")

        if source == ModelSource.TIMM:
            # Use cached model if available, otherwise load it
            if hasattr(self, "_cached_model") and self._cached_model is not None:
                model_for_config = self._cached_model
            else:
                model_for_config = self.load_model(dtype_override=dtype_override)

            # Preprocess image using model's data config
            data_config = resolve_data_config({}, model=model_for_config)
            data_transforms = create_transform(**data_config)
            inputs = data_transforms(image).unsqueeze(0)
        else:
            # Use standard torchvision preprocessing for OSMR
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
            inputs = preprocess(image).unsqueeze(0)

        # Replicate tensors for batch size
        inputs = inputs.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs

    def print_cls_results(self, compiled_model_out):
        """Print classification results.

        Args:
            compiled_model_out: Output from the compiled model
        """
        print_compiled_model_results(compiled_model_out)
