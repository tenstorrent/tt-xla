# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RenderFormer model loader implementation.

RenderFormer is a transformer-based neural renderer for triangle meshes
with global illumination. Reference: https://github.com/microsoft/renderformer
"""

import torch
from typing import Optional

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


class ModelVariant(StrEnum):
    """Available RenderFormer model variants."""

    V1_1_SWIN_LARGE = "v1.1_Swin_Large"


class ModelLoader(ForgeModel):
    """RenderFormer model loader implementation."""

    _VARIANTS = {
        ModelVariant.V1_1_SWIN_LARGE: ModelConfig(
            pretrained_model_name="microsoft/renderformer-v1.1-swin-large",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V1_1_SWIN_LARGE

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
            model="RenderFormer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the RenderFormer model from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The RenderFormer model instance.
        """
        from renderformer.models.renderformer import RenderFormer

        model_name = self._variant_config.pretrained_model_name
        model = RenderFormer.from_pretrained(model_name, **kwargs)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the RenderFormer model.

        Creates dummy triangle mesh inputs suitable for the RenderFormer
        rendering pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Dictionary of input tensors with keys: triangles, texture,
                  mask, vn, c2w, fov.
        """
        num_triangles = 128
        tex_patch_size = 32
        num_views = 1

        triangles = torch.randn(batch_size, num_triangles, 3, 3)
        texture = torch.randn(
            batch_size, num_triangles, 13, tex_patch_size, tex_patch_size
        )
        mask = torch.ones(batch_size, num_triangles, dtype=torch.bool)
        vn = torch.randn(batch_size, num_triangles, 3, 3)
        c2w = (
            torch.eye(4)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, num_views, -1, -1)
            .clone()
        )
        fov = torch.full((batch_size, num_views, 1), 60.0)

        inputs = {
            "triangles": triangles,
            "texture": texture,
            "mask": mask,
            "vn": vn,
            "c2w": c2w,
            "fov": fov,
        }

        if dtype_override is not None:
            for key in inputs:
                if inputs[key].is_floating_point():
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
