# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Orpheus TTS model loader implementation for text-to-speech tasks.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available Orpheus TTS model variants."""

    ORPHEUS_3B_FT = "3b_0.1_ft"


class ModelLoader(ForgeModel):
    """Orpheus TTS model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.ORPHEUS_3B_FT: ModelConfig(
            pretrained_model_name="canopylabs/orpheus-3b-0.1-ft",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ORPHEUS_3B_FT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Orpheus TTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

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

        # Orpheus TTS prompt format: "<voice>: <text>"
        prompt = "tara: Hello, this is a test of the Orpheus text to speech model."
        inputs = self.tokenizer(prompt, return_tensors="pt")
        return inputs
