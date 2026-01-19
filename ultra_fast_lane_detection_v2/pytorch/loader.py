# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ultra-Fast-Lane-Detection-v2 model loader implementation
"""

import torch
import os
from typing import Optional
from dataclasses import dataclass

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
from ...tools.utils import get_file


@dataclass
class LaneDetectionV2Config(ModelConfig):
    """Configuration specific to Ultra-Fast-Lane-Detection-v2 models"""

    input_height: int  # Model input height
    input_width: int  # Model input width


class ModelVariant(StrEnum):
    """Available Ultra-Fast-Lane-Detection-v2 model variants."""

    TUSIMPLE_RESNET34 = "tusimple_resnet34"


class ModelLoader(ForgeModel):
    """Ultra-Fast-Lane-Detection-v2 model loader implementation."""

    _VARIANTS = {
        ModelVariant.TUSIMPLE_RESNET34: LaneDetectionV2Config(
            pretrained_model_name="tusimple_res34",
            input_height=320,
            input_width=800,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TUSIMPLE_RESNET34

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
    ):
        """
        Initialize the Ultra-Fast-Lane-Detection-v2 model loader.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.config: LaneDetectionV2Config = self._variant_config

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
            model="ultra-fast-lane-detection-v2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            source=ModelSource.GITHUB,
            task=ModelTask.CV_IMAGE_SEG,
            framework=Framework.TORCH,
        )

    def load_model(
        self, dtype_override: Optional[torch.dtype] = None
    ) -> torch.nn.Module:
        """
        Load the Ultra-Fast-Lane-Detection-v2 TuSimple34 model.

        Args:
            dtype_override: Optional torch.dtype override (default: float32).

        Returns:
            torch.nn.Module: Loaded TuSimple34 model
        """
        from .src.utils import load_model

        model = load_model(
            model_path=get_file(
                "test_files/pytorch/Ultrafast_lane_detection_v2/tusimple_res34.pth"
            ),
            input_height=self.config.input_height,
            input_width=self.config.input_width,
        )
        model.eval()

        # Convert model dtype if requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(
        self, batch_size: int = 1, dtype_override: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """
        Generate random input tensor for the model.

        Args:
            batch_size: Batch size for the input tensor. Default: 1
            dtype_override: Optional torch.dtype override (default: float32).

        Returns:
            torch.Tensor: Random input tensor
        """
        # Generate random input tensor with specified batch size directly
        inputs = torch.randn(
            batch_size,
            3,
            self.config.input_height,
            self.config.input_width,
            dtype=torch.float32,
        )

        # Convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
