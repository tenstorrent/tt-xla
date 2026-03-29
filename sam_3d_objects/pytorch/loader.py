# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SAM-3D-Objects model loader implementation for 3D object reconstruction.

Loads the SparseStructureEncoder from the SAM-3D pipeline, which encodes
3D input into a latent representation for downstream 3D reconstruction.

Requires the sam-3d-objects repository to be cloned at /tmp/sam3d_objects_repo.
"""
import os
import sys

import torch
import yaml
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
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

SAM3D_REPO_PATH = "/tmp/sam3d_objects_repo"


def _ensure_sam3d_importable():
    """Ensure the sam-3d-objects repo is cloned and importable."""
    if not os.path.isdir(SAM3D_REPO_PATH):
        import subprocess

        subprocess.check_call(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--recurse-submodules",
                "https://github.com/facebookresearch/sam-3d-objects.git",
                SAM3D_REPO_PATH,
            ]
        )
    if SAM3D_REPO_PATH not in sys.path:
        sys.path.insert(0, SAM3D_REPO_PATH)


class ModelVariant(StrEnum):
    """Available SAM-3D-Objects model variants."""

    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    """SAM-3D-Objects model loader for the SparseStructureEncoder."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="facebook/sam-3d-objects",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    _CKPT_NAME = "checkpoints/ss_encoder"
    _RESOLUTION = 16
    _IN_CHANNELS = 8

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SAM-3D-Objects",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SAM-3D SparseStructureEncoder.

        Returns:
            torch.nn.Module: The sparse structure encoder model.
        """
        _ensure_sam3d_importable()
        from sam3d_objects.model.backbone.tdfy_dit.models.sparse_structure_vae import (
            SparseStructureEncoder,
        )

        repo_id = self._variant_config.pretrained_model_name

        config_path = hf_hub_download(repo_id, f"{self._CKPT_NAME}.yaml")
        weights_path = hf_hub_download(repo_id, f"{self._CKPT_NAME}.safetensors")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Remove hydra _target_ and other meta keys before passing as kwargs
        model_args = {k: v for k, v in config.items() if not k.startswith("_")}
        model = SparseStructureEncoder(**model_args)

        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the SparseStructureEncoder.

        Returns:
            torch.Tensor: A 3D volume tensor [B, C, D, H, W].
        """
        dtype = dtype_override or torch.float32

        # x: 3D volume input [B, C, D, H, W]
        x = torch.randn(
            batch_size,
            self._IN_CHANNELS,
            self._RESOLUTION,
            self._RESOLUTION,
            self._RESOLUTION,
            dtype=dtype,
        )

        return x
