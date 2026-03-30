# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SpeechJudge-GRM model loader implementation for speech naturalness evaluation.
"""
import torch
from transformers import (
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniProcessor,
)
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from .src.model import Wrapper


class ModelVariant(StrEnum):
    """Available SpeechJudge-GRM model variants."""

    SPEECHJUDGE_GRM_7B = "7B"


class ModelLoader(ForgeModel):
    """SpeechJudge-GRM model loader for speech naturalness evaluation.

    This model is a Generative Reward Model fine-tuned from Qwen2.5-Omni-7B
    for evaluating speech naturalness in TTS outputs.
    """

    _VARIANTS = {
        ModelVariant.SPEECHJUDGE_GRM_7B: LLMModelConfig(
            pretrained_model_name="RMSnow/SpeechJudge-GRM",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SPEECHJUDGE_GRM_7B

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Evaluate the naturalness of this speech."},
            ],
        }
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""
        return ModelInfo(
            model="SpeechJudge-GRM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load processor for the current variant."""
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SpeechJudge-GRM model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"low_cpu_mem_usage": True}

        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.float32
        model_kwargs |= kwargs

        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.config.use_cache = False
        model.eval()
        model = Wrapper(model)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the SpeechJudge-GRM model."""
        if self.processor is None:
            self._load_processor()

        text = self.processor.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )

        return inputs
