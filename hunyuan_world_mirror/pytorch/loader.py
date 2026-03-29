# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HunyuanWorld-Mirror model loader implementation for 3D geometric prediction.

Loads the WorldMirror model, a feed-forward architecture for universal 3D world
reconstruction from images. It performs camera pose estimation, depth prediction,
point cloud generation, surface normal estimation, and 3D Gaussian generation.

Requires the HunyuanWorld-Mirror repository to be cloned at /tmp/hunyuan_world_mirror_repo.
"""
import os
import sys

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

REPO_PATH = "/tmp/hunyuan_world_mirror_repo"


def _ensure_repo_importable():
    """Ensure the HunyuanWorld-Mirror repo is cloned and importable."""
    if REPO_PATH not in sys.path:
        if not os.path.isdir(REPO_PATH):
            import subprocess

            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror.git",
                    REPO_PATH,
                ]
            )

        sys.path.insert(0, REPO_PATH)


class ModelVariant(StrEnum):
    """Available HunyuanWorld-Mirror model variants."""

    WORLD_MIRROR = "World_Mirror"


class ModelLoader(ForgeModel):
    """HunyuanWorld-Mirror model loader for universal 3D geometric prediction."""

    _VARIANTS = {
        ModelVariant.WORLD_MIRROR: ModelConfig(
            pretrained_model_name="tencent/HunyuanWorld-Mirror",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.WORLD_MIRROR

    # Model architecture constants
    _IMG_SIZE = 518
    _NUM_VIEWS = 2  # Minimum number of input views for 3D reconstruction

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="HunyuanWorld-Mirror",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the WorldMirror model.

        Returns:
            torch.nn.Module: The WorldMirror 3D geometric prediction model.
        """
        _ensure_repo_importable()
        from src.models.models.worldmirror import WorldMirror

        repo_id = self._variant_config.pretrained_model_name
        model = WorldMirror.from_pretrained(repo_id)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the WorldMirror model.

        Returns:
            dict: Input dict with 'views' and 'cond_flags' for the model forward pass.
        """
        dtype = dtype_override or torch.float32

        # img: input images [B, N, 3, H, W] in [0, 1]
        img = torch.rand(
            batch_size,
            self._NUM_VIEWS,
            3,
            self._IMG_SIZE,
            self._IMG_SIZE,
            dtype=dtype,
        )

        views = {"img": img}
        cond_flags = [0, 0, 0]  # No optional priors: [camera_pose, depth, intrinsics]

        return {"views": views, "cond_flags": cond_flags}
