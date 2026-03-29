# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MDLM (Masked Diffusion Language Model) loader implementation.
"""
import torch
from typing import Optional

from transformers import AutoModelForMaskedLM, AutoTokenizer
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available MDLM model variants."""

    MDLM_OWT = "OWT"


class ModelLoader(ForgeModel):
    """MDLM model loader implementation."""

    _VARIANTS = {
        ModelVariant.MDLM_OWT: LLMModelConfig(
            pretrained_model_name="kuleshov-group/mdlm-owt",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MDLM_OWT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MDLM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(
            pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            "The quick brown fox jumps over the lazy dog.",
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        timesteps = torch.zeros(1)

        return {
            "input_ids": inputs["input_ids"],
            "timesteps": timesteps,
        }

    def decode_output(self, outputs):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        predicted_ids = logits.argmax(dim=-1)
        decoded = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        print("Decoded output:", decoded)
        return decoded
