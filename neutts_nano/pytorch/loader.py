# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NeuTTS-Nano model loader implementation for text-to-speech tasks.
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
    """Available NeuTTS-Nano model variants."""

    NEUTTS_NANO = "Nano"


class ModelLoader(ForgeModel):
    """NeuTTS-Nano model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.NEUTTS_NANO: ModelConfig(
            pretrained_model_name="neuphonic/neutts-nano",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEUTTS_NANO

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="NeuTTS-Nano",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype_override or torch.float32,
            trust_remote_code=True,
            **kwargs,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        input_ids = torch.randint(0, 1000, (1, 64))
        attention_mask = torch.ones(1, 64, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
