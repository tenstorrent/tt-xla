# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SD3.5 Large GGUF model loader implementation.

Loads the calcuis/sd3.5-large-gguf GGUF-format SD3 transformer model using
diffusers' GGUF quantization support. This is a quantized version of
stabilityai/stable-diffusion-3.5-large for efficient storage and inference.

Available variants:
- Q4_0: 4-bit quantized SD3.5 Large transformer (4.77GB)
"""

from typing import Any, Optional

import torch
from diffusers import GGUFQuantizationConfig, SD3Transformer2DModel
from huggingface_hub import hf_hub_download

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

REPO_ID = "calcuis/sd3.5-large-gguf"

# GGUF filename for Q4_0 quantization
Q4_0_FILENAME = "sd3.5_large-q4_0.gguf"

# SD3.5 Large transformer config source
TRANSFORMER_CONFIG = "stabilityai/stable-diffusion-3.5-large"
TRANSFORMER_SUBFOLDER = "transformer"

# SD3.5 Large transformer input dimensions
LATENT_CHANNELS = 16
LATENT_HEIGHT = 64
LATENT_WIDTH = 64
JOINT_ATTENTION_DIM = 4096
POOLED_PROJECTION_DIM = 2048
MAX_SEQ_LEN = 154


class ModelVariant(StrEnum):
    """Available SD3.5 Large GGUF model variants."""

    Q4_0 = "q4_0"


class ModelLoader(ForgeModel):
    """SD3.5 Large GGUF model loader for quantized SD3 transformer."""

    _VARIANTS = {
        ModelVariant.Q4_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.Q4_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SD3.5_Large_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the SD3.5 Large transformer from GGUF format.

        Returns:
            SD3Transformer2DModel instance loaded from GGUF checkpoint.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._transformer is None:
            gguf_path = hf_hub_download(REPO_ID, Q4_0_FILENAME)
            self._transformer = SD3Transformer2DModel.from_single_file(
                gguf_path,
                config=TRANSFORMER_CONFIG,
                subfolder=TRANSFORMER_SUBFOLDER,
                quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
                torch_dtype=dtype,
            )
            self._transformer.eval()
        elif dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare synthetic inputs for the SD3.5 transformer.

        Returns:
            dict: Input tensors matching the SD3Transformer2DModel forward signature:
                - hidden_states: Latent tensor [batch, channels, height, width]
                - timestep: Scalar timestep tensor
                - encoder_hidden_states: Text encoder outputs [batch, seq_len, dim]
                - pooled_projections: Pooled text embeddings [batch, pooled_dim]
        """
        dtype = kwargs.get("dtype_override", torch.float32)
        return {
            "hidden_states": torch.randn(
                1, LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH, dtype=dtype
            ),
            "timestep": torch.tensor([1.0], dtype=dtype),
            "encoder_hidden_states": torch.randn(
                1, MAX_SEQ_LEN, JOINT_ATTENTION_DIM, dtype=dtype
            ),
            "pooled_projections": torch.randn(1, POOLED_PROJECTION_DIM, dtype=dtype),
        }
