# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Hunyuan3D-2mini model loader implementation for image-to-3D generation.

Loads the Hunyuan3DDiT (DiT backbone) from the Hunyuan3D-2mini pipeline,
which is the core flow-matching diffusion transformer for generating 3D
shapes from image conditioning.

Requires the Hunyuan3D-2 repository to be cloned at /tmp/hunyuan3d2_repo.
"""
import os
import sys
import types

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

HUNYUAN3D_REPO_PATH = "/tmp/hunyuan3d2_repo"


def _ensure_hunyuan3d_importable():
    """Ensure the Hunyuan3D-2 repo is cloned and importable."""
    if "hy3dgen" not in sys.modules:
        if not os.path.isdir(HUNYUAN3D_REPO_PATH):
            import subprocess

            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "https://github.com/Tencent/Hunyuan3D-2.git",
                    HUNYUAN3D_REPO_PATH,
                ]
            )

        sys.path.insert(0, HUNYUAN3D_REPO_PATH)
        hy3dgen_mod = types.ModuleType("hy3dgen")
        sys.modules["hy3dgen"] = hy3dgen_mod
        hy3dgen_mod.__path__ = [os.path.join(HUNYUAN3D_REPO_PATH, "hy3dgen")]


class ModelVariant(StrEnum):
    """Available Hunyuan3D-2mini model variants."""

    HUNYUAN3D_DIT_V2_MINI = "hunyuan3d_dit_v2_mini"


class ModelLoader(ForgeModel):
    """Hunyuan3D-2mini model loader for the Hunyuan3DDiT (DiT backbone)."""

    _VARIANTS = {
        ModelVariant.HUNYUAN3D_DIT_V2_MINI: ModelConfig(
            pretrained_model_name="tencent/Hunyuan3D-2mini",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUNYUAN3D_DIT_V2_MINI

    # Model config from hunyuan3d-dit-v2-mini/config.yaml
    _SUBFOLDER = "hunyuan3d-dit-v2-mini"
    _IN_CHANNELS = 64
    _NUM_LATENTS = 512
    _CONTEXT_IN_DIM = 1536
    # DINOv2 ViT-g/14: image_size=518, patch_size=14 -> 37*37=1369 patches + 1 CLS
    _COND_SEQ_LEN = 1370

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="hunyuan3d_2mini",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Hunyuan3DDiT model.

        Returns:
            torch.nn.Module: The DiT flow-matching backbone.
        """
        _ensure_hunyuan3d_importable()
        from hy3dgen.shapegen.models.denoisers import Hunyuan3DDiT

        repo_id = self._variant_config.pretrained_model_name

        config_path = hf_hub_download(repo_id, f"{self._SUBFOLDER}/config.yaml")
        weights_path = hf_hub_download(
            repo_id, f"{self._SUBFOLDER}/model.fp16.safetensors"
        )

        with open(config_path) as f:
            config = yaml.safe_load(f)

        model_params = config["model"]["params"]
        model = Hunyuan3DDiT(**model_params)

        state_dict = load_file(weights_path)
        # Weights are stored under the 'model' key in the checkpoint
        model_state = {
            k.removeprefix("model."): v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }
        if model_state:
            model.load_state_dict(model_state)
        else:
            model.load_state_dict(state_dict)

        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the Hunyuan3DDiT model.

        Returns:
            dict: Input tensors (x, t, contexts) for the model forward pass.
        """
        dtype = dtype_override or torch.float32

        # x: noisy latent tokens [B, num_latents, in_channels]
        x = torch.randn(
            batch_size,
            self._NUM_LATENTS,
            self._IN_CHANNELS,
            dtype=dtype,
        )

        # t: diffusion timestep
        t = torch.full((batch_size,), 0.5, dtype=dtype)

        # contexts: DINOv2 image conditioning tokens
        contexts = {
            "main": torch.randn(
                batch_size,
                self._COND_SEQ_LEN,
                self._CONTEXT_IN_DIM,
                dtype=dtype,
            )
        }

        return {"x": x, "t": t, "contexts": contexts}
