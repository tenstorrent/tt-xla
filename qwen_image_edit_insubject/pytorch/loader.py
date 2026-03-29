# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen-Image-Edit-InSubject LoRA model loader implementation.

Loads the Qwen-Image-Edit diffusion transformer and applies the
peteromallet/Qwen-Image-Edit-InSubject LoRA adapter for improved
subject preservation during image editing.

Available variants:
- QWEN_IMAGE_EDIT_INSUBJECT: InSubject LoRA (bf16)
"""

from typing import Any, Optional

import torch
from diffusers import QwenImageEditPipeline

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

BASE_MODEL = "Qwen/Qwen-Image-Edit"
LORA_REPO = "peteromallet/Qwen-Image-Edit-InSubject"
LORA_WEIGHT_NAME = "InSubject-0.5.safetensors"


class ModelVariant(StrEnum):
    """Available Qwen-Image-Edit-InSubject model variants."""

    QWEN_IMAGE_EDIT_INSUBJECT = "Edit_InSubject"


class ModelLoader(ForgeModel):
    """Qwen-Image-Edit-InSubject LoRA model loader for subject preservation."""

    _VARIANTS = {
        ModelVariant.QWEN_IMAGE_EDIT_INSUBJECT: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.QWEN_IMAGE_EDIT_INSUBJECT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="QWEN_IMAGE_EDIT_INSUBJECT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load the Qwen-Image-Edit transformer with InSubject LoRA weights applied.

        Loads the full pipeline, applies the diffusers-format LoRA, fuses the
        weights into the transformer, and returns the transformer component.

        Returns:
            QwenImageTransformer2DModel with LoRA weights fused.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        pipe = QwenImageEditPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=dtype,
        )
        pipe.load_lora_weights(
            LORA_REPO,
            weight_name=LORA_WEIGHT_NAME,
        )
        pipe.fuse_lora()

        self._transformer = pipe.transformer
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
