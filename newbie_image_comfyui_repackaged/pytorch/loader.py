# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NewBie-image ComfyUI Repackaged model loader implementation.

Loads text encoder components from Comfy-Org/NewBie-image-Exp0.1_repackaged.
The NewBie diffusion pipeline uses dual text encoders:
- Gemma 3 4B IT (language model for text conditioning)
- Jina CLIP v2 (vision-language model for text/image conditioning)

Text encoders are loaded from their upstream pretrained repositories since
the ComfyUI-repackaged single-file safetensors lack standard diffusers configs.

Available variants:
- GEMMA_3_TEXT_ENCODER: Gemma 3 4B IT text encoder (google/gemma-3-4b-it)
- JINA_CLIP_V2_TEXT_ENCODER: Jina CLIP v2 text encoder (jinaai/jina-clip-v2)
"""

from typing import Any, Optional

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)

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

REPO_ID = "Comfy-Org/NewBie-image-Exp0.1_repackaged"

# Upstream pretrained repos for text encoder components
GEMMA_REPO = "google/gemma-3-4b-it"
JINA_CLIP_REPO = "jinaai/jina-clip-v2"


class ModelVariant(StrEnum):
    """Available NewBie-image ComfyUI Repackaged model variants."""

    GEMMA_3_TEXT_ENCODER = "Gemma3_TextEncoder"
    JINA_CLIP_V2_TEXT_ENCODER = "JinaCLIPv2_TextEncoder"


class ModelLoader(ForgeModel):
    """NewBie-image ComfyUI Repackaged model loader for text encoder components."""

    _VARIANTS = {
        ModelVariant.GEMMA_3_TEXT_ENCODER: ModelConfig(
            pretrained_model_name=GEMMA_REPO,
        ),
        ModelVariant.JINA_CLIP_V2_TEXT_ENCODER: ModelConfig(
            pretrained_model_name=JINA_CLIP_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.GEMMA_3_TEXT_ENCODER

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._model = None
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="NEWBIE_IMAGE_COMFYUI_REPACKAGED",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_gemma_encoder(
        self, dtype: torch.dtype = torch.float32
    ) -> AutoModelForCausalLM:
        """Load Gemma 3 4B IT as a text encoder."""
        self._tokenizer = AutoTokenizer.from_pretrained(GEMMA_REPO)
        self._model = AutoModelForCausalLM.from_pretrained(
            GEMMA_REPO,
            torch_dtype=dtype,
            use_cache=False,
        )
        self._model.eval()
        return self._model

    def _load_jina_clip_encoder(self, dtype: torch.dtype = torch.float32) -> AutoModel:
        """Load Jina CLIP v2 as a text encoder."""
        self._tokenizer = AutoTokenizer.from_pretrained(
            JINA_CLIP_REPO, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            JINA_CLIP_REPO,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self._model.eval()
        return self._model

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the text encoder model for the selected variant.

        Returns:
            The text encoder model instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        if self._model is None:
            if self._variant == ModelVariant.GEMMA_3_TEXT_ENCODER:
                return self._load_gemma_encoder(dtype)
            return self._load_jina_clip_encoder(dtype)
        if dtype_override is not None:
            self._model = self._model.to(dtype=dtype_override)
        return self._model

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the text encoder.

        Returns tokenized text inputs appropriate for the selected variant.
        """
        dtype = kwargs.get("dtype_override", torch.float32)

        if self._tokenizer is None:
            if self._variant == ModelVariant.GEMMA_3_TEXT_ENCODER:
                self._tokenizer = AutoTokenizer.from_pretrained(GEMMA_REPO)
            else:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    JINA_CLIP_REPO, trust_remote_code=True
                )

        prompt = "A photo of an astronaut riding a horse on mars"

        if self._variant == ModelVariant.GEMMA_3_TEXT_ENCODER:
            tokens = self._tokenizer(
                prompt,
                return_tensors="pt",
                max_length=64,
                padding="max_length",
                truncation=True,
            )
            return {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
            }

        # Jina CLIP v2 text input
        tokens = self._tokenizer(
            prompt,
            return_tensors="pt",
            max_length=64,
            padding="max_length",
            truncation=True,
        )
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }
