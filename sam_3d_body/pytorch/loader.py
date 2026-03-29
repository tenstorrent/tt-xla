# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SAM 3D Body (DINOv3) model loader implementation for 3D human mesh recovery.

SAM 3D Body estimates complete 3D pose and shape of the human body, including
feet and hands, from a single image using the Momentum Human Rig (MHR)
representation.
"""

import torch
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available SAM 3D Body model variants."""

    DINOV3 = "DINOv3"


class ModelLoader(ForgeModel):
    """SAM 3D Body model loader implementation for 3D human mesh recovery."""

    _VARIANTS = {
        ModelVariant.DINOV3: ModelConfig(
            pretrained_model_name="facebook/sam-3d-body-dinov3",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DINOV3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SAM 3D Body",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_KEYPOINT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from sam_3d_body import load_sam_3d_body_hf

        repo_id = self._variant_config.pretrained_model_name
        model, _ = load_sam_3d_body_hf(repo_id, device="cpu")
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        # SAM 3D Body expects RGB image tensors of shape (batch, 3, H, W)
        # normalized to [0, 1] range
        pixel_values = torch.rand(batch_size, 3, 512, 512)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return {"pixel_values": pixel_values}
