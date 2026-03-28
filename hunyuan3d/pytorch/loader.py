# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Hunyuan3D model loader implementation for image-to-3D generation.

Loads the HunYuanDiTPlain (DiT backbone) from the Hunyuan3D-2.1 pipeline,
which is the core flow-matching diffusion transformer for generating 3D shapes
from image conditioning.

Requires the Hunyuan3D-2.1 repository to be cloned at /tmp/hunyuan3d_repo.
"""
import os
import sys
import types
from typing import Optional

import torch
import yaml
from huggingface_hub import hf_hub_download

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

HUNYUAN3D_REPO_PATH = "/tmp/hunyuan3d_repo"


def _ensure_hunyuan3d_importable():
    """Ensure the Hunyuan3D-2.1 shape repo is cloned and importable."""
    if "hy3dshape" not in sys.modules:
        if not os.path.isdir(HUNYUAN3D_REPO_PATH):
            import subprocess

            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git",
                    HUNYUAN3D_REPO_PATH,
                ]
            )

        shape_pkg = os.path.join(HUNYUAN3D_REPO_PATH, "hy3dshape")
        if shape_pkg not in sys.path:
            sys.path.insert(0, shape_pkg)

        hy3dshape_mod = types.ModuleType("hy3dshape")
        sys.modules["hy3dshape"] = hy3dshape_mod
        hy3dshape_mod.__path__ = [os.path.join(shape_pkg, "hy3dshape")]


class ModelVariant(StrEnum):
    """Available Hunyuan3D model variants."""

    DIT_V2_1 = "DiT_v2_1"


class ModelLoader(ForgeModel):
    """Hunyuan3D model loader for the HunYuanDiTPlain (DiT backbone)."""

    _VARIANTS = {
        ModelVariant.DIT_V2_1: ModelConfig(
            pretrained_model_name="tencent/Hunyuan3D-2.1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DIT_V2_1

    # Architecture parameters from config.yaml
    _NUM_LATENTS = 4096
    _IN_CHANNELS = 64
    _CONTEXT_DIM = 1024
    _TEXT_LEN = 1370  # DINOv2 ViT-L/14 @ 518px: (518/14)^2 + 1 = 1370

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Hunyuan3D",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Hunyuan3D DiT model.

        Returns:
            torch.nn.Module: The HunYuanDiTPlain diffusion transformer.
        """
        _ensure_hunyuan3d_importable()
        from hy3dshape.models.denoisers.hunyuandit import HunYuanDiTPlain

        repo_id = self._variant_config.pretrained_model_name

        config_path = hf_hub_download(repo_id, "hunyuan3d-dit-v2-1/config.yaml")
        weights_path = hf_hub_download(repo_id, "hunyuan3d-dit-v2-1/model.fp16.ckpt")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        model_params = config["model"]["params"]

        model = HunYuanDiTPlain(**model_params)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the HunYuanDiTPlain model.

        Returns:
            dict: Input tensors (x, t, contexts) for the model forward pass.
        """
        dtype = dtype_override or torch.float32

        # x: latent tokens [B, num_latents, in_channels]
        x = torch.randn(
            batch_size,
            self._NUM_LATENTS,
            self._IN_CHANNELS,
            dtype=dtype,
        )

        # t: diffusion timestep [B]
        t = torch.full((batch_size,), 0.5, dtype=dtype)

        # contexts: DINOv2 image conditioning tokens
        contexts = {
            "main": torch.randn(
                batch_size,
                self._TEXT_LEN,
                self._CONTEXT_DIM,
                dtype=dtype,
            ),
        }

        return {"x": x, "t": t, "contexts": contexts}
