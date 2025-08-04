# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Swin model loader implementation
"""

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
from torchvision import models
import torch
from PIL import Image
from torchvision import models
from ...tools.utils import get_file, print_compiled_model_results
from typing import Optional
from dataclasses import dataclass


@dataclass
class SwinConfig(ModelConfig):
    """Configuration specific to Swin models"""

    model_name: str = None
    weight_name: str = None


class ModelVariant(StrEnum):
    """Available Swin model variants."""

    SWIN_T = "swin_t"
    SWIN_S = "swin_s"
    SWIN_B = "swin_b"
    SWIN_V2_T = "swin_v2_t"
    SWIN_V2_S = "swin_v2_s"
    SWIN_V2_B = "swin_v2_b"


class ModelLoader(ForgeModel):

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.SWIN_T: SwinConfig(
            pretrained_model_name="swin_t",
            model_name="swin_t",
            weight_name="Swin_T_Weights",
        ),
        ModelVariant.SWIN_S: SwinConfig(
            pretrained_model_name="swin_s",
            model_name="swin_s",
            weight_name="Swin_S_Weights",
        ),
        ModelVariant.SWIN_B: SwinConfig(
            pretrained_model_name="swin_b",
            model_name="swin_b",
            weight_name="Swin_B_Weights",
        ),
        ModelVariant.SWIN_V2_T: SwinConfig(
            pretrained_model_name="swin_v2_t",
            model_name="swin_v2_t",
            weight_name="Swin_V2_T_Weights",
        ),
        ModelVariant.SWIN_V2_S: SwinConfig(
            pretrained_model_name="swin_v2_s",
            model_name="swin_v2_s",
            weight_name="Swin_V2_S_Weights",
        ),
        ModelVariant.SWIN_V2_B: SwinConfig(
            pretrained_model_name="swin_v2_b",
            model_name="swin_v2_b",
            weight_name="Swin_V2_B_Weights",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.SWIN_T

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional variant. If None, uses DEFAULT_VARIANT.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="swin",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.TORCH_HUB,
            framework=Framework.TORCH,
        )

    """Loads Swin model and sample input."""

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters - no longer hardcoded
        self._weights = None

    def load_model(self, dtype_override=None):
        """Load pretrained Swin model."""
        model_name = self._variant_config.model_name
        weight_name = self._variant_config.weight_name

        weights = getattr(models, weight_name).DEFAULT
        model = getattr(models, model_name)(weights=weights)
        model.eval()

        self._weights = weights

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for Swin model"""

        preprocess = self._weights.transforms()
        image_file = get_file("http://images.cocodataset.org/val2017/000000039769.jpg")
        image = Image.open(image_file).convert("RGB")
        img_t = preprocess(image)
        batch_t = torch.unsqueeze(img_t, 0)
        inputs = batch_t.contiguous()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs

    def print_cls_results(self, compiled_model_out):
        print_compiled_model_results(compiled_model_out)
