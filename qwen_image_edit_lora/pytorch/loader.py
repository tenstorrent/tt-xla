# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-Edit-2511 Multiple Angles LoRA model loader implementation.

Loads the Qwen-Image-Edit-2511 base diffusion transformer and applies
the fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA weights for multi-angle
camera-controlled image editing from a single input image.
"""

from typing import Any, Optional

import torch
from diffusers import QwenImageTransformer2DModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

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

BASE_MODEL = "Qwen/Qwen-Image-Edit-2511"
LORA_REPO = "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA"
LORA_FILENAME = "qwen-image-edit-2511-multiple-angles-lora.safetensors"


class ModelVariant(StrEnum):
    """Available Qwen-Image-Edit LoRA variants."""

    MULTIPLE_ANGLES = "Multiple_Angles"


class ModelLoader(ForgeModel):
    """Qwen-Image-Edit-2511 Multiple Angles LoRA model loader."""

    _VARIANTS = {
        ModelVariant.MULTIPLE_ANGLES: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.MULTIPLE_ANGLES

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer: Optional[QwenImageTransformer2DModel] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_IMAGE_EDIT_LORA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load the Qwen-Image-Edit-2511 transformer with LoRA weights applied."""
        dtype = dtype_override if dtype_override is not None else torch.float32

        # Load base diffusion transformer
        model_kwargs = {
            "subfolder": "transformer",
            "torch_dtype": dtype,
        }
        model_kwargs |= kwargs
        self._transformer = QwenImageTransformer2DModel.from_pretrained(
            BASE_MODEL,
            **model_kwargs,
        )

        # Download and apply LoRA weights
        lora_path = hf_hub_download(
            repo_id=LORA_REPO,
            filename=LORA_FILENAME,
        )
        lora_state_dict = load_file(lora_path)
        self._transformer.load_attn_procs(lora_state_dict)

        self._transformer.eval()
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the diffusion transformer.

        Returns a dict matching QwenImageTransformer2DModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        batch_size = kwargs.get("batch_size", 1)

        # From model config: in_channels=64 (img_in linear input dimension)
        img_dim = 64
        # joint_attention_dim from config = 3584
        text_dim = 3584
        txt_seq_len = 32

        # img_seq_len must equal frame * height * width for positional encoding
        frame, height, width = 1, 8, 8
        img_seq_len = frame * height * width

        hidden_states = torch.randn(batch_size, img_seq_len, img_dim, dtype=dtype)
        encoder_hidden_states = torch.randn(
            batch_size, txt_seq_len, text_dim, dtype=dtype
        )
        encoder_hidden_states_mask = torch.ones(batch_size, txt_seq_len, dtype=dtype)
        timestep = torch.tensor([500.0] * batch_size, dtype=dtype)
        img_shapes = [(frame, height, width)] * batch_size

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }
