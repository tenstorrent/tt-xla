# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kimi-Audio model loader implementation for audio understanding and generation tasks.
"""
import torch
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


class ModelVariant(StrEnum):
    """Available Kimi-Audio model variants."""

    KIMI_AUDIO_7B_INSTRUCT = "7B_Instruct"


class ModelLoader(ForgeModel):
    """Kimi-Audio model loader implementation for audio understanding and generation tasks."""

    _VARIANTS = {
        ModelVariant.KIMI_AUDIO_7B_INSTRUCT: ModelConfig(
            pretrained_model_name="moonshotai/Kimi-Audio-7B-Instruct",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KIMI_AUDIO_7B_INSTRUCT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="KimiAudio",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Kimi-Audio model instance."""
        from transformers import AutoModelForCausalLM

        pretrained_model_name = self._variant_config.pretrained_model_name

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            torch_dtype=dtype_override if dtype_override is not None else torch.float32,
            **kwargs,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Kimi-Audio model."""
        from transformers import AutoTokenizer

        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
        )

        prompt = "Please transcribe the following audio."
        inputs = tokenizer(prompt, return_tensors="pt")

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
