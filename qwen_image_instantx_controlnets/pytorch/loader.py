# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image InstantX ControlNets model loader implementation.

Loads FluxControlNetModel variants from the Comfy-Org/Qwen-Image-InstantX-ControlNets
repository. Supports Inpainting and Union ControlNet variants.

Available variants:
- INPAINTING: ControlNet for image inpainting
- UNION: ControlNet supporting multiple control modes (canny, tile, depth, blur, pose, gray, low-quality)
"""

from typing import Any, Optional

import torch
from diffusers import FluxControlNetModel
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

REPO_ID = "Comfy-Org/Qwen-Image-InstantX-ControlNets"

_CONTROLNET_FILES = {
    "inpainting": "Qwen-Image-InstantX-ControlNet-Inpainting.safetensors",
    "union": "Qwen-Image-InstantX-ControlNet-Union.safetensors",
}

# Union ControlNet supports 7 modes: canny(0), tile(1), depth(2), blur(3), pose(4), gray(5), low-quality(6)
_UNION_NUM_MODES = 7


class ModelVariant(StrEnum):
    """Available Qwen-Image InstantX ControlNet model variants."""

    INPAINTING = "Inpainting"
    UNION = "Union"


class ModelLoader(ForgeModel):
    """Qwen-Image InstantX ControlNets model loader."""

    _VARIANTS = {
        ModelVariant.INPAINTING: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.UNION: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.UNION

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._controlnet = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_IMAGE_INSTANTX_CONTROLNETS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _get_version_key(self) -> str:
        """Map variant to version key for file lookup."""
        return {
            ModelVariant.INPAINTING: "inpainting",
            ModelVariant.UNION: "union",
        }[self._variant]

    def _load_controlnet(
        self, dtype: torch.dtype = torch.float32
    ) -> FluxControlNetModel:
        """Load ControlNet from single-file safetensors."""
        version = self._get_version_key()

        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=_CONTROLNET_FILES[version],
        )

        # Build config kwargs based on variant
        config_kwargs = {}
        if self._variant == ModelVariant.UNION:
            config_kwargs["num_mode"] = _UNION_NUM_MODES

        self._controlnet = FluxControlNetModel(**config_kwargs)

        state_dict = load_file(model_path)
        self._controlnet.load_state_dict(state_dict)
        self._controlnet.to(dtype=dtype)
        self._controlnet.eval()
        return self._controlnet

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the ControlNet model.

        Returns:
            FluxControlNetModel instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._controlnet is None:
            return self._load_controlnet(dtype)
        if dtype_override is not None:
            self._controlnet = self._controlnet.to(dtype=dtype_override)
        return self._controlnet

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the ControlNet.

        Returns a dict matching FluxControlNetModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        batch_size = kwargs.get("batch_size", 1)

        # FluxControlNetModel default config values
        in_channels = 64
        joint_attention_dim = 4096
        pooled_projection_dim = 768

        # Sequence lengths for image and text tokens
        img_seq_len = 64  # e.g. 8x8 patch grid
        txt_seq_len = 32

        hidden_states = torch.randn(batch_size, img_seq_len, in_channels, dtype=dtype)
        controlnet_cond = torch.randn(batch_size, img_seq_len, in_channels, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, joint_attention_dim, dtype=dtype
        )
        pooled_projections = torch.randn(batch_size, pooled_projection_dim, dtype=dtype)
        timestep = torch.tensor([500.0] * batch_size, dtype=dtype)
        img_ids = torch.zeros(img_seq_len, 3, dtype=dtype)
        txt_ids = torch.zeros(txt_seq_len, 3, dtype=dtype)

        inputs = {
            "hidden_states": hidden_states,
            "controlnet_cond": controlnet_cond,
            "encoder_hidden_states": encoder_hidden_states,
            "pooled_projections": pooled_projections,
            "timestep": timestep,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
        }

        # Union variant requires controlnet_mode
        if self._variant == ModelVariant.UNION:
            # Use mode 0 (canny) as default
            inputs["controlnet_mode"] = torch.zeros(batch_size, dtype=torch.long)

        return inputs
