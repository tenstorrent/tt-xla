# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Orpheus-TTS model loader implementation for text-to-speech tasks.
"""
import torch
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM

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


class ModelVariant(StrEnum):
    """Available Orpheus-TTS model variants."""

    ORPHEUS_3B_DE_FT = "3B-De-Ft"


class ModelLoader(ForgeModel):
    """Orpheus-TTS model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.ORPHEUS_3B_DE_FT: ModelConfig(
            pretrained_model_name="canopylabs/3b-de-ft-research_release",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ORPHEUS_3B_DE_FT

    sample_text = "Hallo, wie geht es Ihnen heute?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Orpheus-TTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(self.sample_text, return_tensors="pt")
        return inputs
