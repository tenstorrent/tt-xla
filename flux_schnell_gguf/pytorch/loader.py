# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.1-schnell GGUF (leejet/FLUX.1-schnell-gguf) model loader implementation.

FLUX.1-schnell is a 12B parameter text-to-image generation model in GGUF quantized format,
based on the FLUX transformer architecture from Black Forest Labs.

Available variants:
- FLUX1_SCHNELL_Q4_0: Q4_0 quantized variant (~6.88 GB)
"""

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
from .src.model_utils import load_flux_gguf_pipe, flux_schnell_preprocessing

REPO_ID = "leejet/FLUX.1-schnell-gguf"
BASE_MODEL = "black-forest-labs/FLUX.1-schnell"


class ModelVariant(StrEnum):
    """Available FLUX.1-schnell GGUF model variants."""

    FLUX1_SCHNELL_Q4_0 = "flux1_schnell_Q4_0"


class ModelLoader(ForgeModel):
    """FLUX.1-schnell GGUF model loader implementation."""

    _VARIANTS = {
        ModelVariant.FLUX1_SCHNELL_Q4_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FLUX1_SCHNELL_Q4_0

    GGUF_FILE = "flux1-schnell-Q4_0.gguf"

    prompt = "An astronaut riding a green horse"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX.1-schnell GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FLUX.1-schnell transformer from GGUF checkpoint.

        Returns:
            torch.nn.Module: The FLUX transformer model instance.
        """
        if self.pipeline is None:
            self.pipeline = load_flux_gguf_pipe(REPO_ID, self.GGUF_FILE, BASE_MODEL)

        if dtype_override is not None:
            self.pipeline.transformer = self.pipeline.transformer.to(dtype_override)

        return self.pipeline.transformer

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the model.

        Returns:
            dict: Input tensors for the FLUX transformer model.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else None

        return flux_schnell_preprocessing(self.pipeline, self.prompt, dtype=dtype)
