# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Granite Speech 3.3 model loader implementation.

Granite Speech 3.3-8B is a multimodal speech-language model from IBM that
combines a Conformer-based speech encoder with a Granite 3.3-8B LLM backbone
for automatic speech recognition (ASR) and speech translation (AST).
"""

from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

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
    """Available Granite Speech 3.3 model variants."""

    V8B = "8B"


class GraniteSpeechWrapper(torch.nn.Module):
    """Wrapper around GraniteSpeechForConditionalGeneration for a clean forward pass."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self, input_ids, attention_mask, input_features, feature_attention_mask
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )


class ModelLoader(ForgeModel):
    """Granite Speech 3.3 model loader implementation."""

    _VARIANTS = {
        ModelVariant.V8B: ModelConfig(
            pretrained_model_name="ibm-granite/granite-speech-3.3-8b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V8B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Granite_Speech_3_3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Granite Speech 3.3 model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            pretrained_model_name,
            device_map="cpu",
            **model_kwargs,
        )
        model.eval()

        self._processor = AutoProcessor.from_pretrained(pretrained_model_name)

        return GraniteSpeechWrapper(model)

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Granite Speech 3.3 model."""
        if self._processor is None:
            self.load_model(dtype_override=dtype_override)

        tokenizer = self._processor.tokenizer

        # Build a chat prompt with the <|audio|> placeholder
        chat = [
            {"role": "user", "content": "<|audio|>Transcribe the speech."},
        ]
        prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        # Generate a synthetic 1-second mono audio clip at 16 kHz
        audio = np.random.randn(16000).astype(np.float32)

        inputs = self._processor(prompt, audio, return_tensors="pt")

        return [
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["input_features"],
            inputs["feature_attention_mask"],
        ]
