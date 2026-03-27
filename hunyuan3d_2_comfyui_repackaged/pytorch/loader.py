# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Hunyuan3D 2.0 ComfyUI Repackaged model loader implementation.

Loads single-file safetensors DiT models from Comfy-Org/hunyuan3D_2.0_repackaged.
Supports the Hunyuan3DDiT denoiser backbone for 3D shape generation.

Available variants:
- DIT_V2: Hunyuan3D DiT v2 base model
- DIT_V2_MV: Hunyuan3D DiT v2 multi-view model
- DIT_V2_MV_TURBO: Hunyuan3D DiT v2 multi-view turbo model

Requires the Hunyuan3D-2 repository to be cloned at /tmp/hunyuan3d2_repo.
"""

import os
import sys
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

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

REPO_ID = "Comfy-Org/hunyuan3D_2.0_repackaged"
HUNYUAN3D_REPO_PATH = "/tmp/hunyuan3d2_repo"

# Default Hunyuan3DDiT architecture parameters
IN_CHANNELS = 64
CONTEXT_IN_DIM = 1536
HIDDEN_SIZE = 1024
NUM_HEADS = 16
DEPTH = 16
DEPTH_SINGLE_BLOCKS = 32

# Test input dimensions
LATENT_SEQ_LEN = 256
COND_SEQ_LEN = 257


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
                    "https://github.com/tencent/Hunyuan3D-2.git",
                    HUNYUAN3D_REPO_PATH,
                ]
            )

        if HUNYUAN3D_REPO_PATH not in sys.path:
            sys.path.insert(0, HUNYUAN3D_REPO_PATH)


class ModelVariant(StrEnum):
    """Available Hunyuan3D 2.0 ComfyUI Repackaged model variants."""

    DIT_V2 = "dit_v2"
    DIT_V2_MV = "dit_v2_mv"
    DIT_V2_MV_TURBO = "dit_v2_mv_turbo"


# Map variants to their safetensors filenames in the HF repo
_VARIANT_FILENAMES = {
    ModelVariant.DIT_V2: "split_files/hunyuan3d-dit-v2_fp16.safetensors",
    ModelVariant.DIT_V2_MV: "split_files/hunyuan3d-dit-v2-mv_fp16.safetensors",
    ModelVariant.DIT_V2_MV_TURBO: "split_files/hunyuan3d-dit-v2-mv-turbo_fp16.safetensors",
}


class ModelLoader(ForgeModel):
    """Hunyuan3D 2.0 ComfyUI Repackaged model loader using single-file safetensors."""

    _VARIANTS = {
        ModelVariant.DIT_V2: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.DIT_V2_MV: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.DIT_V2_MV_TURBO: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.DIT_V2

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="HUNYUAN3D_2_COMFYUI_REPACKAGED",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Hunyuan3DDiT model.

        Returns:
            Hunyuan3DDiT: The DiT denoiser model for 3D shape generation.
        """
        _ensure_hunyuan3d_importable()
        from hy3dgen.shapegen.models.denoisers import Hunyuan3DDiT

        variant = self._variant or self.DEFAULT_VARIANT
        filename = _VARIANT_FILENAMES[variant]

        weights_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
        state_dict = load_file(weights_path)

        model = Hunyuan3DDiT(
            in_channels=IN_CHANNELS,
            context_in_dim=CONTEXT_IN_DIM,
            hidden_size=HIDDEN_SIZE,
            num_heads=NUM_HEADS,
            depth=DEPTH,
            depth_single_blocks=DEPTH_SINGLE_BLOCKS,
        )
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

        # x: noisy 3D shape latent [B, N, in_channels]
        x = torch.randn(batch_size, LATENT_SEQ_LEN, IN_CHANNELS, dtype=dtype)

        # t: diffusion timestep [B]
        t = torch.full((batch_size,), 0.5, dtype=dtype)

        # contexts: image conditioning dict with 'main' key [B, seq_len, context_in_dim]
        cond = torch.randn(batch_size, COND_SEQ_LEN, CONTEXT_IN_DIM, dtype=dtype)

        return {"x": x, "t": t, "contexts": {"main": cond}}
