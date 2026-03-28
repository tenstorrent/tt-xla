# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FireRed-Image-Edit-1.0 GGUF model loader implementation.

Loads a GGUF-quantized diffusion transformer for instruction-guided image
editing. The model is based on the Qwen-Image-Edit architecture (20B params)
and accepts a source image plus a text prompt describing the desired edit.

Available variants:
- FIRERED_Q4_0: FireRed-Image-Edit-1.0 Q4_0 quantization
"""

from typing import Any, Optional

import torch
from diffusers import GGUFQuantizationConfig, QwenImageTransformer2DModel

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

REPO_ID = "Arunk25/FireRed-Image-Edit-1.0_comfy_GGUF"
CONFIG_REPO = "Qwen/Qwen-Image-Edit"


class ModelVariant(StrEnum):
    """Available FireRed-Image-Edit GGUF model variants."""

    FIRERED_Q4_0 = "FireRed_Q4_0"


class ModelLoader(ForgeModel):
    """FireRed-Image-Edit-1.0 GGUF model loader."""

    _VARIANTS = {
        ModelVariant.FIRERED_Q4_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.FIRERED_Q4_0

    GGUF_FILE = "firered_image_edit_1.0_q4_0.gguf"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FireRed_Image_Edit_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the FireRed-Image-Edit GGUF diffusion transformer.

        Returns:
            QwenImageTransformer2DModel instance.
        """
        compute_dtype = dtype_override if dtype_override is not None else torch.bfloat16
        quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

        repo_id = self._variant_config.pretrained_model_name
        self._transformer = QwenImageTransformer2DModel.from_single_file(
            f"https://huggingface.co/{repo_id}/resolve/main/{self.GGUF_FILE}",
            quantization_config=quantization_config,
            config=CONFIG_REPO,
            subfolder="transformer",
            torch_dtype=compute_dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the diffusion transformer.

        Returns a dict matching QwenImageTransformer2DModel.forward() signature.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
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
