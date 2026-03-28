# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX.1-dev LoRA model loader implementation.

Loads the black-forest-labs/FLUX.1-dev base pipeline and applies style LoRA
weights from alexrzem/flux-loras for stylized text-to-image generation.

Available variants:
- FLUX_LORA_PIXAR_3D: Pixar-style 3D rendering LoRA
- FLUX_LORA_COMIC_BOOK: Comic book style LoRA
"""

from typing import Any, Optional

import torch
from diffusers import FluxPipeline

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

BASE_MODEL = "black-forest-labs/FLUX.1-dev"
LORA_REPO = "alexrzem/flux-loras"

# LoRA weight filenames (under dev/ subdirectory in the repo)
LORA_PIXAR_3D = "dev/Pixar_3D_-_leimaxiu252537.safetensors"
LORA_COMIC_BOOK = "dev/Comic_Book_v4_-_Adel_AI.safetensors"


class ModelVariant(StrEnum):
    """Available FLUX LoRA style variants."""

    FLUX_LORA_PIXAR_3D = "FLUX_LoRA_Pixar_3D"
    FLUX_LORA_COMIC_BOOK = "FLUX_LoRA_Comic_Book"


_LORA_FILES = {
    ModelVariant.FLUX_LORA_PIXAR_3D: LORA_PIXAR_3D,
    ModelVariant.FLUX_LORA_COMIC_BOOK: LORA_COMIC_BOOK,
}


class ModelLoader(ForgeModel):
    """FLUX.1-dev LoRA model loader."""

    _VARIANTS = {
        ModelVariant.FLUX_LORA_PIXAR_3D: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
        ModelVariant.FLUX_LORA_COMIC_BOOK: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.FLUX_LORA_PIXAR_3D

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[FluxPipeline] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="FLUX_LORAS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(
        self,
        *,
        dtype_override: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """Load the FLUX.1-dev pipeline with style LoRA weights applied.

        Returns:
            FluxPipeline with LoRA weights merged.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = FluxPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        lora_file = _LORA_FILES[self._variant]
        self.pipeline.load_lora_weights(
            LORA_REPO,
            weight_name=lora_file,
        )

        return self.pipeline

    def load_inputs(self, prompt: Optional[str] = None, **kwargs) -> Any:
        """Prepare inputs for text-to-image generation.

        Returns:
            dict with prompt key.
        """
        if prompt is None:
            prompt = "An astronaut riding a green horse"

        return {
            "prompt": prompt,
        }
