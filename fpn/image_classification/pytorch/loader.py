# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FPN (Feature Pyramid Network) model loader implementation
"""
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available FPN model variants."""

    RESNET50_FPN_V2 = "resnet50_fpn_v2"


class FPNWrapper(nn.Module):
    """FPN wrapper model that extracts just the FPN component from FasterRCNN."""

    def __init__(self):
        super().__init__()

        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
        self.fpn = model.backbone.fpn

    def forward(self, feat0, feat1, feat2):
        x = OrderedDict()
        x["feat0"] = feat0
        x["feat1"] = feat1
        x["feat2"] = feat2
        outputs = self.fpn(x)
        outputs = tuple(outputs.values())
        return outputs


class ModelLoader(ForgeModel):
    """FPN model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.RESNET50_FPN_V2: ModelConfig(
            pretrained_model_name="torchvision/fasterrcnn_resnet50_fpn_v2",
        )
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.RESNET50_FPN_V2

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
            model="fpn",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.TORCHVISION,
            framework=Framework.TORCH,
        )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    def load_model(self, dtype_override=None):
        """Load and return the FPN model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype.

        Returns:
            torch.nn.Module: The FPN model instance.
        """
        model = FPNWrapper()

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        return model

    def load_inputs(self, batch_size=1, dtype_override=None):
        """Generate sample inputs for the FPN model.

        Args:
            batch_size: Number of samples in the batch
            dtype_override: Optional torch.dtype to override input dtype

        Returns:
            List of input tensors matching the expected FPN input format
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        feat0 = torch.rand(batch_size, 256, 64, 64, dtype=dtype)
        feat1 = torch.rand(batch_size, 512, 16, 16, dtype=dtype)
        feat2 = torch.rand(batch_size, 2048, 8, 8, dtype=dtype)

        return [feat0, feat1, feat2]
