# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TRELLIS model loader implementation for image-to-3D generation.

Loads the SparseStructureFlowModel (DiT backbone) from the TRELLIS pipeline,
which is the core flow-matching transformer for generating 3D sparse structures
from image conditioning.

Requires the TRELLIS repository to be cloned at /tmp/trellis_repo.
"""
import json
import os
import sys

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

# TRELLIS requires the repo on sys.path and SDPA attention backend
# (flash_attn/xformers are CUDA-only)
TRELLIS_REPO_PATH = "/tmp/trellis_repo"
os.environ.setdefault("ATTN_BACKEND", "sdpa")


def _ensure_trellis_importable():
    """Ensure the TRELLIS repo is cloned and importable."""
    import types

    if "trellis" not in sys.modules:
        if not os.path.isdir(TRELLIS_REPO_PATH):
            import subprocess

            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--recurse-submodules",
                    "https://github.com/microsoft/TRELLIS.git",
                    TRELLIS_REPO_PATH,
                ]
            )

        sys.path.insert(0, TRELLIS_REPO_PATH)
        trellis_mod = types.ModuleType("trellis")
        sys.modules["trellis"] = trellis_mod
        trellis_mod.__path__ = [os.path.join(TRELLIS_REPO_PATH, "trellis")]


class ModelVariant(StrEnum):
    """Available TRELLIS model variants."""

    SPARSE_STRUCTURE_FLOW = "Sparse_Structure_Flow"


class ModelLoader(ForgeModel):
    """TRELLIS model loader for the SparseStructureFlowModel (DiT backbone)."""

    _VARIANTS = {
        ModelVariant.SPARSE_STRUCTURE_FLOW: ModelConfig(
            pretrained_model_name="microsoft/TRELLIS-image-large",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SPARSE_STRUCTURE_FLOW

    # Model config from ss_flow_img_dit_L_16l8_fp16.json
    _CKPT_NAME = "ckpts/ss_flow_img_dit_L_16l8_fp16"
    _RESOLUTION = 16
    _IN_CHANNELS = 8
    _COND_CHANNELS = 1024
    _COND_SEQ_LEN = 257  # DINOv2 ViT-L/14: 1 CLS + 256 patch tokens

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="TRELLIS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the TRELLIS SparseStructureFlowModel.

        Returns:
            torch.nn.Module: The sparse structure flow model (DiT backbone).
        """
        _ensure_trellis_importable()
        from trellis.models.sparse_structure_flow import SparseStructureFlowModel

        repo_id = self._variant_config.pretrained_model_name

        config_path = hf_hub_download(repo_id, f"{self._CKPT_NAME}.json")
        weights_path = hf_hub_download(repo_id, f"{self._CKPT_NAME}.safetensors")

        with open(config_path) as f:
            config = json.load(f)

        args = config["args"]
        args["use_fp16"] = False

        model = SparseStructureFlowModel(**args)
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the SparseStructureFlowModel.

        Returns:
            dict: Input tensors (x, t, cond) for the model forward pass.
        """
        dtype = dtype_override or torch.float32

        # x: noisy 3D volume latent [B, C, D, H, W]
        x = torch.randn(
            batch_size,
            self._IN_CHANNELS,
            self._RESOLUTION,
            self._RESOLUTION,
            self._RESOLUTION,
            dtype=dtype,
        )

        # t: diffusion timestep
        t = torch.full((batch_size,), 0.5, dtype=dtype)

        # cond: DINOv2 image conditioning tokens [B, seq_len, cond_channels]
        cond = torch.randn(
            batch_size,
            self._COND_SEQ_LEN,
            self._COND_CHANNELS,
            dtype=dtype,
        )

        return {"x": x, "t": t, "cond": cond}
