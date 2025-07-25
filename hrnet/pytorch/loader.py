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
from PIL import Image

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
from ...tools.utils import get_file, print_compiled_model_results


class ModelVariant(StrEnum):
    """Available HRNet model variants."""

    HRNET_W18_SMALL = "hrnet_w18_small"
    HRNET_W18_SMALL_V2 = "hrnet_w18_small_v2"
    HRNET_W18 = "hrnet_w18"
    HRNET_W30 = "hrnet_w30"
    HRNET_W32 = "hrnet_w32"
    HRNET_W40 = "hrnet_w40"
    HRNET_W44 = "hrnet_w44"
    HRNET_W48 = "hrnet_w48"
    HRNET_W64 = "hrnet_w64"
    HRNET_W18_MS_AUG_IN1K = "hrnet_w18.ms_aug_in1k"


class ModelLoader(ForgeModel):
    """HRNet model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.HRNET_W18_SMALL: ModelConfig(
            pretrained_model_name="hrnet_w18_small",
        ),
        ModelVariant.HRNET_W18_SMALL_V2: ModelConfig(
            pretrained_model_name="hrnet_w18_small_v2",
        ),
        ModelVariant.HRNET_W18: ModelConfig(
            pretrained_model_name="hrnet_w18",
        ),
        ModelVariant.HRNET_W30: ModelConfig(
            pretrained_model_name="hrnet_w30",
        ),
        ModelVariant.HRNET_W32: ModelConfig(
            pretrained_model_name="hrnet_w32",
        ),
        ModelVariant.HRNET_W40: ModelConfig(
            pretrained_model_name="hrnet_w40",
        ),
        ModelVariant.HRNET_W44: ModelConfig(
            pretrained_model_name="hrnet_w44",
        ),
        ModelVariant.HRNET_W48: ModelConfig(
            pretrained_model_name="hrnet_w48",
        ),
        ModelVariant.HRNET_W64: ModelConfig(
            pretrained_model_name="hrnet_w64",
        ),
        ModelVariant.HRNET_W18_MS_AUG_IN1K: ModelConfig(
            pretrained_model_name="hrnet_w18.ms_aug_in1k",
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
        return ModelInfo(
            model="hrnet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_KEYPOINT_DET,
            source=ModelSource.TIMM,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load pretrained HRNet model for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The HRNet model instance.
        """
        # Get the pretrained model name from the instance's variant config
        model_name = self._variant_config.pretrained_model_name

        # Load model using timm
        model = timm.create_model(model_name, pretrained=True)
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
        # Get the Image
        image_file = get_file(
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
        )
        image = Image.open(image_file)

        # Use cached model if available, otherwise load it
        if hasattr(self, "_cached_model") and self._cached_model is not None:
            model_for_config = self._cached_model
        else:
            model_for_config = self.load_model(dtype_override)

        # Preprocess image using model's data config
        data_config = resolve_data_config({}, model=model_for_config)
        transforms = create_transform(**data_config)
        inputs = transforms(image).unsqueeze(0)

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
