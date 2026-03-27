# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Hunyuan3D model loader implementation for 3D asset generation.

Loads the DiT (Diffusion Transformer) backbone from the Hunyuan3D-2 pipeline,
which is the core flow-matching transformer for generating 3D shapes from
image conditioning.

Based on Comfy-Org/hunyuan3D_2.1_repackaged (originally tencent/Hunyuan3D-2).

Requires the Hunyuan3D-2 repository to be cloned at /tmp/hunyuan3d_repo.
"""
import os
import sys
import types

import torch
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

HUNYUAN3D_REPO_PATH = "/tmp/hunyuan3d_repo"


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
                    "--recurse-submodules",
                    "https://github.com/Tencent/Hunyuan3D-2.git",
                    HUNYUAN3D_REPO_PATH,
                ]
            )

        sys.path.insert(0, HUNYUAN3D_REPO_PATH)
        hy3dgen_mod = types.ModuleType("hy3dgen")
        sys.modules["hy3dgen"] = hy3dgen_mod
        hy3dgen_mod.__path__ = [os.path.join(HUNYUAN3D_REPO_PATH, "hy3dgen")]


class ModelVariant(StrEnum):
    """Available Hunyuan3D model variants."""

    DIT_V2 = "DiT_v2"


class ModelLoader(ForgeModel):
    """Hunyuan3D model loader for the DiT flow-matching backbone."""

    _VARIANTS = {
        ModelVariant.DIT_V2: ModelConfig(
            pretrained_model_name="tencent/Hunyuan3D-2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DIT_V2

    # DiT checkpoint path within the HF repo
    _CKPT_SUBFOLDER = "hunyuan3d-dit-v2-0"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Hunyuan3D",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Hunyuan3D DiT flow-matching model.

        Returns:
            torch.nn.Module: The DiT backbone for 3D shape generation.
        """
        _ensure_hunyuan3d_importable()
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        repo_id = self._variant_config.pretrained_model_name
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(repo_id)
        model = pipeline.model
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the DiT flow-matching model.

        The DiT processes noisy 3D volume latents conditioned on image
        embeddings, using a flow-matching timestep schedule.

        Returns:
            dict: Input tensors (x, t, cond) for the model forward pass.
        """
        dtype = dtype_override or torch.float32

        # x: noisy 3D volume latent [B, C, D, H, W]
        # Shape based on Hunyuan3D-2 VAE latent configuration
        x = torch.randn(batch_size, 64, 8, 8, 8, dtype=dtype)

        # t: flow-matching timestep in [0, 1]
        t = torch.full((batch_size,), 0.5, dtype=dtype)

        # cond: image conditioning embeddings from CLIP conditioner
        # [B, seq_len, hidden_dim]
        cond = torch.randn(batch_size, 257, 1024, dtype=dtype)

        return {"x": x, "t": t, "cond": cond}
