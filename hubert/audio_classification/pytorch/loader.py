# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
HuBERT model loader implementation for audio classification (speech emotion recognition).
"""

from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available HuBERT audio classification model variants."""

    BASE_CH_SPEECH_EMOTION = "Base_CH_Speech_Emotion"


class ModelLoader(ForgeModel):
    """HuBERT model loader implementation for audio classification (PyTorch)."""

    _VARIANTS = {
        ModelVariant.BASE_CH_SPEECH_EMOTION: ModelConfig(
            pretrained_model_name="xmj2002/hubert-base-ch-speech-emotion-recognition",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_CH_SPEECH_EMOTION

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="HuBERT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        from transformers import Wav2Vec2FeatureExtractor

        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["dtype"] = dtype_override

        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self._variant_config.pretrained_model_name, **processor_kwargs
        )

        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        import torch
        import torch.nn as nn
        from transformers import HubertPreTrainedModel, HubertModel

        class HubertClassificationHead(nn.Module):
            """Classification head for speech emotion recognition."""

            def __init__(self, config):
                super().__init__()
                self.dense = nn.Linear(config.hidden_size, config.hidden_size)
                self.dropout = nn.Dropout(config.final_dropout)
                self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

            def forward(self, features, **kwargs):
                x = features
                x = self.dropout(x)
                x = self.dense(x)
                x = torch.tanh(x)
                x = self.dropout(x)
                x = self.out_proj(x)
                return x

        class HubertForSpeechClassification(HubertPreTrainedModel):
            """HuBERT-based speech emotion classifier."""

            def __init__(self, config):
                super().__init__(config)
                self.hubert = HubertModel(config)
                self.classifier = HubertClassificationHead(config)
                self.post_init()

            def forward(self, input_values):
                outputs = self.hubert(input_values)
                hidden_states = outputs[0]
                hidden_states = torch.mean(hidden_states, dim=1)
                logits = self.classifier(hidden_states)
                return logits

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = HubertForSpeechClassification.from_pretrained(
            self._variant_config.pretrained_model_name,
            **model_kwargs,
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np

        if self._processor is None:
            self._load_processor(dtype_override=dtype_override)

        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self._processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        return inputs
