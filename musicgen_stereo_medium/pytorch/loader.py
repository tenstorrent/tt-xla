# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MusicGen Stereo Medium model loader implementation for stereo music generation.

Loads facebook/musicgen-stereo-medium, a 1.5B parameter auto-regressive
Transformer that generates stereo music from text descriptions.
"""

from typing import Optional

import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

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


class ModelVariant(StrEnum):
    """Available MusicGen Stereo Medium model variants."""

    MUSICGEN_STEREO_MEDIUM = "MusicGen_Stereo_Medium"


class ModelLoader(ForgeModel):
    """MusicGen Stereo Medium model loader implementation for stereo music generation."""

    _VARIANTS = {
        ModelVariant.MUSICGEN_STEREO_MEDIUM: ModelConfig(
            pretrained_model_name="facebook/musicgen-stereo-medium",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MUSICGEN_STEREO_MEDIUM

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MusicGen Stereo Medium",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        return self.model

    def load_inputs(self, *, dtype_override=None, **kwargs):
        if self.processor is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.processor(
            text=[
                "80s pop track with bassy drums and synth",
                "90s rock song with loud guitars and heavy drums",
            ],
            padding=True,
            return_tensors="pt",
        )

        pad_token_id = self.model.generation_config.pad_token_id
        decoder_input_ids = (
            torch.ones(
                (inputs.input_ids.shape[0] * self.model.decoder.num_codebooks, 1),
                dtype=torch.long,
            )
            * pad_token_id
        )

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "decoder_input_ids": decoder_input_ids,
        }
