# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DUSt3R model loader implementation for dense 3D reconstruction from stereo image pairs.

DUSt3R (Dense and Unconstrained Stereo 3D Reconstruction) uses an asymmetric
encoder-decoder transformer to predict dense 3D point maps from image pairs
without requiring camera calibration.

Requires the dust3r repository to be cloned at /tmp/dust3r_repo.
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

DUST3R_REPO_PATH = "/tmp/dust3r_repo"


def _ensure_dust3r_importable():
    """Ensure the dust3r repo is cloned and importable."""
    if not os.path.isdir(DUST3R_REPO_PATH):
        import subprocess

        subprocess.check_call(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--recurse-submodules",
                "https://github.com/naver/dust3r.git",
                DUST3R_REPO_PATH,
            ]
        )

    if DUST3R_REPO_PATH not in sys.path:
        sys.path.insert(0, DUST3R_REPO_PATH)


class ModelVariant(StrEnum):
    """Available DUSt3R model variants."""

    VITLARGE_BASEDECODER_512_DPT = "ViTLarge_BaseDecoder_512_dpt"


class ModelLoader(ForgeModel):
    """DUSt3R model loader for dense stereo 3D reconstruction."""

    _VARIANTS = {
        ModelVariant.VITLARGE_BASEDECODER_512_DPT: ModelConfig(
            pretrained_model_name="naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VITLARGE_BASEDECODER_512_DPT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DUSt3R",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_DEPTH_EST,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DUSt3R AsymmetricCroCo3DStereo model.

        Returns:
            torch.nn.Module: The DUSt3R stereo 3D reconstruction model.
        """
        _ensure_dust3r_importable()
        from dust3r.model import AsymmetricCroCo3DStereo

        repo_id = self._variant_config.pretrained_model_name
        model = AsymmetricCroCo3DStereo.from_pretrained(repo_id)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample stereo image pair inputs for DUSt3R.

        DUSt3R expects a list of dicts, each with 'img' and 'true_shape' keys.
        The model processes pairs of images to produce 3D point maps.

        Returns:
            list: Two-element list of dicts with image tensors and shape info.
        """
        dtype = dtype_override or torch.float32
        height, width = 384, 512

        view1 = {
            "img": torch.randn(batch_size, 3, height, width, dtype=dtype),
            "true_shape": torch.tensor([[height, width]] * batch_size),
        }
        view2 = {
            "img": torch.randn(batch_size, 3, height, width, dtype=dtype),
            "true_shape": torch.tensor([[height, width]] * batch_size),
        }

        return [view1, view2]
