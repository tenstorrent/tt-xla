# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image DiffSynth ControlNets model loader implementation.

Loads single-file safetensors ControlNet variants from
Comfy-Org/Qwen-Image-DiffSynth-ControlNets. These are blockwise ControlNet
models for conditioning Qwen-Image generation with canny edges, depth maps,
or inpainting masks.

Available variants:
- CANNY:   Canny edge ControlNet
- DEPTH:   Depth map ControlNet
- INPAINT: Inpainting ControlNet (has extra condition channels)
"""

from typing import Any, Optional

import torch
from diffusers import QwenImageControlNetModel
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

REPO_ID = "Comfy-Org/Qwen-Image-DiffSynth-ControlNets"

_CONTROLNET_FILES = {
    "canny": "split_files/model_patches/qwen_image_canny_diffsynth_controlnet.safetensors",
    "depth": "split_files/model_patches/qwen_image_depth_diffsynth_controlnet.safetensors",
    "inpaint": "split_files/model_patches/qwen_image_inpaint_diffsynth_controlnet.safetensors",
}


class ModelVariant(StrEnum):
    """Available Qwen-Image DiffSynth ControlNet model variants."""

    CANNY = "Canny"
    DEPTH = "Depth"
    INPAINT = "Inpaint"


class ModelLoader(ForgeModel):
    """Qwen-Image DiffSynth ControlNets model loader."""

    _VARIANTS = {
        ModelVariant.CANNY: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.DEPTH: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.INPAINT: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.CANNY

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._controlnet = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_IMAGE_DIFFSYNTH_CONTROLNETS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _get_variant_key(self) -> str:
        return {
            ModelVariant.CANNY: "canny",
            ModelVariant.DEPTH: "depth",
            ModelVariant.INPAINT: "inpaint",
        }[self._variant]

    def _load_controlnet(
        self, dtype: torch.dtype = torch.float32
    ) -> QwenImageControlNetModel:
        variant_key = self._get_variant_key()

        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=_CONTROLNET_FILES[variant_key],
        )

        extra_condition_channels = 4 if variant_key == "inpaint" else 0
        self._controlnet = QwenImageControlNetModel(
            extra_condition_channels=extra_condition_channels,
        )

        state_dict = load_file(model_path)
        self._controlnet.load_state_dict(state_dict)
        self._controlnet = self._controlnet.to(dtype=dtype)
        self._controlnet.eval()
        return self._controlnet

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._controlnet is None:
            return self._load_controlnet(dtype)
        if dtype_override is not None:
            self._controlnet = self._controlnet.to(dtype=dtype_override)
        return self._controlnet

    def load_inputs(self, **kwargs) -> Any:
        dtype = kwargs.get("dtype_override", torch.float32)
        batch_size = kwargs.get("batch_size", 1)

        in_channels = 64
        text_dim = 3584
        txt_seq_len = 32

        frame, height, width = 1, 8, 8
        img_seq_len = frame * height * width

        variant_key = self._get_variant_key()
        extra_channels = 4 if variant_key == "inpaint" else 0
        controlnet_cond_channels = in_channels + extra_channels

        hidden_states = torch.randn(batch_size, img_seq_len, in_channels, dtype=dtype)
        controlnet_cond = torch.randn(
            batch_size, img_seq_len, controlnet_cond_channels, dtype=dtype
        )
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, text_dim, dtype=dtype
        )
        encoder_hidden_states_mask = torch.ones(batch_size, txt_seq_len, dtype=dtype)
        timestep = torch.tensor([500.0] * batch_size, dtype=dtype)
        img_shapes = [(frame, height, width)] * batch_size

        return {
            "hidden_states": hidden_states,
            "controlnet_cond": controlnet_cond,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }
