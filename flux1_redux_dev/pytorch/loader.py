# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Flux1-Redux-Dev model loader implementation.

Loads the ReduxImageEncoder (image_embedder) component from the
FLUX.1-Redux-dev prior pipeline. This adapter transforms SigLIP vision
encoder hidden states into prompt embeddings compatible with FLUX diffusion.

Available variants:
- FLUX1_REDUX_DEV: Redux image embedder (Comfy-Org/Flux1-Redux-Dev)
"""

from typing import Any, Optional

import torch
from diffusers import FluxPriorReduxPipeline

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

COMFY_REPO_ID = "Comfy-Org/Flux1-Redux-Dev"
BFL_REPO_ID = "black-forest-labs/FLUX.1-Redux-dev"


class ModelVariant(StrEnum):
    """Available Flux1-Redux-Dev model variants."""

    FLUX1_REDUX_DEV = "Redux-Dev"


class ModelLoader(ForgeModel):
    """Flux1-Redux-Dev model loader for the Redux image embedder."""

    _VARIANTS = {
        ModelVariant.FLUX1_REDUX_DEV: ModelConfig(
            pretrained_model_name=COMFY_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.FLUX1_REDUX_DEV

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX1_REDUX_DEV",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype = torch.float32):
        """Load the FluxPriorReduxPipeline from the upstream BFL repo."""
        self._pipe = FluxPriorReduxPipeline.from_pretrained(
            BFL_REPO_ID,
            torch_dtype=dtype,
        )
        return self._pipe

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Redux image embedder.

        Returns the ReduxImageEncoder that transforms SigLIP hidden states
        into FLUX-compatible prompt embeddings.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._pipe is None:
            self._load_pipeline(dtype)
        if dtype_override is not None:
            self._pipe.image_embedder = self._pipe.image_embedder.to(dtype_override)
        return self._pipe.image_embedder

    def load_inputs(self, **kwargs) -> Any:
        """Prepare inputs for the Redux image embedder.

        Creates a random tensor matching SigLIP-SO400M hidden state dimensions.
        The image embedder expects hidden states of shape
        [batch, seq_len, hidden_size] from the penultimate layer of SigLIP.
        """
        dtype = kwargs.get("dtype_override", torch.float32)

        if self._pipe is None:
            self._load_pipeline(dtype)

        # Get dimensions from the loaded image encoder config
        image_encoder_config = self._pipe.image_encoder.config
        hidden_size = image_encoder_config.hidden_size
        image_size = image_encoder_config.image_size
        patch_size = image_encoder_config.patch_size
        seq_len = (image_size // patch_size) ** 2

        hidden_states = torch.randn(1, seq_len, hidden_size, dtype=dtype)
        return hidden_states
